//! Phase two (draft 3 Def. 6.8 and §6.7; F1R3Comb v0.6 §5.3–§5.5): eliminate
//! the binder. Each unit — the top-level term, or the process a gate stores,
//! or any quoted process — whose own level binds names is compiled to
//!
//! ```text
//!   (atoms not mentioning its names) | inst(r*, f*, C) | e(f*)
//! ```
//!
//! where the context `C` holds the atoms that do mention them, each bound
//! name written as the leaf `□·γ(j)` beneath the hole. An instance receives
//! its address σ at `r*`; `inst` fills the hole and delivers `@C[σ]` at `f*`;
//! `e(f*)` erects it. Two steps per instance, one premise, one address input
//! at one static channel: Def. 6.10 by construction (Req. 5.26).
//!
//! Units nest: a unit stored inside another's context refers to its parent's
//! names through `□1`, `□2`, … (de Bruijn over context nodes), which is what
//! lets `inst` fill a parent without touching the holes of its children.
//!
//! `Scheme::Positional` compiles every bound name to a static name that is a
//! function of the unit's text — the construction of draft 3 §6.1 — as the
//! negative control Obligations 11.3 and 11.4 require. It is never deployed
//! by the CLI.

use f1r3comb_ir::{IName, Node};
use f1r3comb_term::addr::{f_star, gamma, hole_leaf, leaf, r_star};
use f1r3comb_term::artefact::Scheme;
use f1r3comb_term::{Atom, CName, Hash32, Shape, Term};
use std::collections::HashMap;

#[derive(Clone)]
enum Loc {
    Hole { level: u32, path: Vec<bool> },
    Static(CName),
}

#[derive(Clone, Default)]
struct Env(HashMap<u32, Loc>);

pub struct Phase2 {
    pub scheme: Scheme,
    memo: HashMap<Hash32, CName>,
    /// Units compiled with bound names (wrappers or positional).
    pub units: u64,
}

/// The base of the positional names of a unit: a root-shaped name spelling
/// 64 bits of the unit's hash.
fn positional_root(key: &Hash32) -> CName {
    let mut c = CName::nil();
    for i in 0..64 {
        let bit = (key.0[i / 8] >> (7 - i % 8)) & 1 == 1;
        c = CName::quote(Term::atom(if bit { Atom::k(c) } else { Atom::bl(c, CName::nil()) }));
    }
    CName::quote(Term::atom(Atom::bl(c, CName::nil())))
}

/// Own variables of a unit, in canonical (pre-order) order: those bound by
/// `New` nodes at the unit's level, not beneath a store or a quotation.
fn own_vars(n: &Node, out: &mut Vec<u32>) {
    match n {
        Node::Par(xs) => xs.iter().for_each(|x| own_vars(x, out)),
        Node::New { vars, body } => {
            out.extend(vars.iter().copied());
            own_vars(body, out);
        }
        _ => {}
    }
}

fn flatten<'a>(n: &'a Node, out: &mut Vec<&'a Node>) {
    match n {
        Node::Nil => {}
        Node::Par(xs) => xs.iter().for_each(|x| flatten(x, out)),
        Node::New { body, .. } => flatten(body, out),
        a => out.push(a),
    }
}

fn mentions(n: &Node, vars: &[u32]) -> bool {
    n.free_vars().iter().any(|v| vars.contains(v))
}

impl Phase2 {
    pub fn new(scheme: Scheme) -> Phase2 {
        Phase2 { scheme, memo: HashMap::new(), units: 0 }
    }

    /// The target term of a closed IR process.
    pub fn closed(&mut self, p: &Node) -> Term {
        Term::from_atoms(self.unit(p, &Env::default(), 0))
    }

    fn closed_name(&mut self, p: &Node) -> CName {
        let h = p.hash();
        if let Some(n) = self.memo.get(&h) {
            return n.clone();
        }
        let n = CName::quote(self.closed(p));
        self.memo.insert(h, n.clone());
        n
    }

    fn name(&mut self, n: &IName, env: &Env, depth: u32) -> CName {
        match n {
            IName::Quote(p) => self.closed_name(p),
            IName::Var(x) => match env.0.get(x) {
                Some(Loc::Hole { level, path }) => hole_leaf(depth - 1 - level, path),
                Some(Loc::Static(c)) => c.clone(),
                None => panic!("phase two: unbound variable {x}"),
            },
        }
    }

    fn unit(&mut self, p: &Node, env: &Env, depth: u32) -> Vec<Atom> {
        let mut own = Vec::new();
        own_vars(p, &mut own);
        let mut atoms = Vec::new();
        flatten(p, &mut atoms);
        if own.is_empty() {
            return atoms.into_iter().map(|a| self.atom(a, env, depth)).collect();
        }
        self.units += 1;
        let mut env2 = env.clone();
        match self.scheme {
            Scheme::Positional => {
                let root = positional_root(&p.hash());
                for (j, v) in own.iter().enumerate() {
                    env2.0.insert(*v, Loc::Static(leaf(&root, &gamma(j as u32 + 1))));
                }
                atoms.into_iter().map(|a| self.atom(a, &env2, depth)).collect()
            }
            Scheme::Inst | Scheme::Curried => {
                for (j, v) in own.iter().enumerate() {
                    env2.0.insert(*v, Loc::Hole { level: depth, path: gamma(j as u32 + 1) });
                }
                let mut outside = Vec::new();
                let mut ctx = Vec::new();
                for a in atoms {
                    if mentions(a, &own) {
                        ctx.push(self.atom(a, &env2, depth + 1));
                    } else {
                        outside.push(self.atom(a, env, depth));
                    }
                }
                outside.push(Atom::inst(r_star(), f_star(), Term::from_atoms(ctx)));
                outside.push(Atom::e(f_star()));
                outside
            }
        }
    }

    fn atom(&mut self, a: &Node, env: &Env, depth: u32) -> Atom {
        let Node::Atom { shape, names, store } = a else { unreachable!() };
        let ns: Vec<CName> = names.iter().map(|n| self.name(n, env, depth)).collect();
        match shape {
            Shape::Q => {
                let body = Term::from_atoms(self.unit(store.as_ref().unwrap(), env, depth));
                Atom::q(ns[0].clone(), body)
            }
            s => Atom::new(*s, ns).expect("phase one emits well-formed atoms"),
        }
    }
}

/// A leaf beneath a hole or a root, or a positional name: never an encoding
/// of a source name (Lem. 7.6).
pub fn is_apparatus(n: &CName) -> bool {
    f1r3comb_term::addr::is_address(n) || *n == r_star() || *n == f_star() || {
        let (b, _) = f1r3comb_term::addr::spine(n);
        b.is_hole().is_some()
    }
}
