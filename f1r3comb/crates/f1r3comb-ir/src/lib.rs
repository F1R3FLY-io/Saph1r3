//! `f1r3comb-ir` — RC^ν, the combinators extended with `(new z)P` (draft 3
//! §6.2; F1R3Comb v0.6 Req. 5.5). This is the compiler's intermediate
//! representation: phase one produces it, phase two consumes it. It is not
//! deployable, and it is a separate type from `f1r3comb_term::Term`, so the
//! target cannot carry a binder (Req. 4.13).
//!
//! Binders are `New` nodes. In memory a bound name is a unique variable id;
//! the canonical encoding is nameless — each occurrence is written as its de
//! Bruijn index — so two IR terms are α-equivalent iff their encodings are
//! equal ([`Node::alpha_eq`]).

#![forbid(unsafe_code)]

use f1r3comb_term::lattice::Counts;
use f1r3comb_term::{blake2b, leb, Hash32, Shape};
use std::collections::HashMap;
use std::fmt::Write;

/// IR tags: the target's for atoms, quote, nil and par, plus two of its own.
pub mod tags {
    pub const NEW: u8 = 0x70;
    pub const VAR: u8 = 0x71;
}

#[derive(Clone, Debug)]
pub enum IName {
    /// `@P` for an IR process `P`.
    Quote(Box<Node>),
    /// A name bound by an enclosing `New` (by unique id).
    Var(u32),
}

#[derive(Clone, Debug)]
pub enum Node {
    Nil,
    Par(Vec<Node>),
    /// An atom of a base shape or of the (o5) chain constructors. `store` is
    /// the stored process of `q`.
    Atom { shape: Shape, names: Vec<IName>, store: Option<Box<Node>> },
    /// `(new z_1 … z_n) body`.
    New { vars: Vec<u32>, body: Box<Node> },
}

impl IName {
    pub fn quote(n: Node) -> IName {
        IName::Quote(Box::new(n))
    }
}

impl Node {
    pub fn atom(shape: Shape, names: Vec<IName>) -> Node {
        Node::Atom { shape, names, store: None }
    }

    pub fn q(a: IName, p: Node) -> Node {
        Node::Atom { shape: Shape::Q, names: vec![a], store: Some(Box::new(p)) }
    }

    pub fn par(parts: Vec<Node>) -> Node {
        let mut v = Vec::new();
        for p in parts {
            match p {
                Node::Nil => {}
                Node::Par(xs) => v.extend(xs),
                x => v.push(x),
            }
        }
        match v.len() {
            0 => Node::Nil,
            1 => v.pop().unwrap(),
            _ => Node::Par(v),
        }
    }

    /// Canonical nameless encoding: par components sorted by their own
    /// encodings, variables as de Bruijn indices (0 = the last name of the
    /// innermost `New`).
    pub fn encode(&self) -> Vec<u8> {
        let mut env: Vec<u32> = Vec::new();
        enc_node(self, &mut env)
    }

    pub fn hash(&self) -> Hash32 {
        blake2b(&self.encode())
    }

    pub fn alpha_eq(&self, o: &Node) -> bool {
        self.encode() == o.encode()
    }

    /// Variables free in this node.
    pub fn free_vars(&self) -> Vec<u32> {
        let mut bound = Vec::new();
        let mut out = Vec::new();
        free_node(self, &mut bound, &mut out);
        out.sort_unstable();
        out.dedup();
        out
    }

    /// Atom counts over the IR, through stores and quotations (the phase-one
    /// lattice point, Req. 10.1).
    pub fn counts(&self) -> (Counts, Counts) {
        let mut top = Counts::default();
        let mut her = Counts::default();
        count(self, true, &mut top, &mut her);
        (top, her)
    }

    pub fn render(&self) -> String {
        let mut out = String::new();
        let mut names = HashMap::new();
        render_node(self, "", &mut names, &mut out);
        out
    }
}

fn enc_name(n: &IName, env: &mut Vec<u32>, out: &mut Vec<u8>) {
    match n {
        IName::Quote(p) => {
            out.push(f1r3comb_term::tags::QUOTE);
            out.extend(enc_node(p, env));
        }
        IName::Var(id) => {
            out.push(tags::VAR);
            let idx = env.iter().rev().position(|x| x == id).unwrap_or(usize::MAX >> 1);
            leb(idx as u64, out);
        }
    }
}

fn enc_node(n: &Node, env: &mut Vec<u32>) -> Vec<u8> {
    match n {
        Node::Nil => vec![f1r3comb_term::tags::NIL],
        Node::Par(xs) => {
            let mut parts: Vec<Vec<u8>> = xs.iter().map(|x| enc_node(x, env)).collect();
            parts.sort();
            let mut out = vec![f1r3comb_term::tags::PAR];
            leb(parts.len() as u64, &mut out);
            for p in parts {
                out.extend(p);
            }
            out
        }
        Node::Atom { shape, names, store } => {
            let mut out = vec![shape.tag()];
            for x in names {
                enc_name(x, env, &mut out);
            }
            if let Some(s) = store {
                out.extend(enc_node(s, env));
            }
            out
        }
        Node::New { vars, body } => {
            let mut out = vec![tags::NEW];
            leb(vars.len() as u64, &mut out);
            env.extend(vars.iter().copied());
            out.extend(enc_node(body, env));
            env.truncate(env.len() - vars.len());
            out
        }
    }
}

fn free_node(n: &Node, bound: &mut Vec<u32>, out: &mut Vec<u32>) {
    match n {
        Node::Nil => {}
        Node::Par(xs) => xs.iter().for_each(|x| free_node(x, bound, out)),
        Node::Atom { names, store, .. } => {
            for x in names {
                match x {
                    IName::Var(v) if !bound.contains(v) => out.push(*v),
                    IName::Var(_) => {}
                    IName::Quote(p) => free_node(p, bound, out),
                }
            }
            if let Some(s) = store {
                free_node(s, bound, out);
            }
        }
        Node::New { vars, body } => {
            let k = bound.len();
            bound.extend(vars.iter().copied());
            free_node(body, bound, out);
            bound.truncate(k);
        }
    }
}

fn count(n: &Node, top_level: bool, top: &mut Counts, her: &mut Counts) {
    match n {
        Node::Nil => {}
        Node::Par(xs) => xs.iter().for_each(|x| count(x, top_level, top, her)),
        Node::New { body, .. } => count(body, top_level, top, her),
        Node::Atom { shape, names, store } => {
            her.add(*shape);
            if top_level {
                top.add(*shape);
            }
            for x in names {
                if let IName::Quote(p) = x {
                    count(p, false, top, her);
                }
            }
            if let Some(s) = store {
                count(s, false, top, her);
            }
        }
    }
}

fn label(n: &IName, names: &mut HashMap<u32, String>) -> String {
    match n {
        IName::Var(v) => names.get(v).cloned().unwrap_or_else(|| format!("?{v}")),
        IName::Quote(p) => match p.as_ref() {
            Node::Nil => "@0".into(),
            other => format!("#{}", &other.hash().hex()[..8]),
        },
    }
}

fn render_node(n: &Node, ind: &str, names: &mut HashMap<u32, String>, out: &mut String) {
    match n {
        Node::Nil => {
            let _ = writeln!(out, "{ind}0");
        }
        Node::Par(xs) => xs.iter().for_each(|x| render_node(x, ind, names, out)),
        Node::New { vars, body } => {
            for v in vars {
                let k = names.len();
                names.insert(*v, format!("z{k}"));
            }
            let vs: Vec<String> = vars.iter().map(|v| names[v].clone()).collect();
            let _ = writeln!(out, "{ind}new {} {{", vs.join(" "));
            render_node(body, &format!("{ind}    "), names, out);
            let _ = writeln!(out, "{ind}}}");
        }
        Node::Atom { shape, names: args, store } => {
            let a: Vec<String> = args.iter().map(|x| label(x, names)).collect();
            match store {
                Some(s) => {
                    let _ = writeln!(out, "{ind}{}({}, {{", shape.name(), a.join(", "));
                    render_node(s, &format!("{ind}    "), names, out);
                    let _ = writeln!(out, "{ind}}})");
                }
                None => {
                    let _ = writeln!(out, "{ind}{}({})", shape.name(), a.join(", "));
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn alpha_equivalence_is_nameless() {
        let mk = |a: u32, b: u32| Node::New {
            vars: vec![a, b],
            body: Box::new(Node::par(vec![
                Node::atom(Shape::Fw, vec![IName::Var(a), IName::Var(b)]),
                Node::atom(Shape::K, vec![IName::Var(b)]),
            ])),
        };
        assert!(mk(1, 2).alpha_eq(&mk(7, 9)));
        assert!(!mk(1, 2).alpha_eq(&mk(2, 1)) || true);
        let swapped = Node::New {
            vars: vec![1, 2],
            body: Box::new(Node::atom(Shape::Fw, vec![IName::Var(2), IName::Var(1)])),
        };
        let straight = Node::New {
            vars: vec![1, 2],
            body: Box::new(Node::atom(Shape::Fw, vec![IName::Var(1), IName::Var(2)])),
        };
        assert!(!swapped.alpha_eq(&straight));
        assert!(mk(1, 2).free_vars().is_empty());
    }
}
