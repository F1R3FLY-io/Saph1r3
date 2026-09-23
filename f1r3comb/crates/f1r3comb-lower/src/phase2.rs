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
use crate::{Diag, Severity};
use std::collections::{HashMap, HashSet};

#[derive(Clone)]
enum Loc {
    Hole { level: u32, path: Vec<bool> },
    Static(CName),
}

#[derive(Clone, Default)]
struct Env(HashMap<u32, Loc>);

pub struct Phase2 {
    pub scheme: Scheme,
    /// Variables the curried erection names statically (the gates' `b`).
    pub static_vars: HashSet<u32>,
    pub diags: Vec<Diag>,
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
        Phase2 { scheme, static_vars: HashSet::new(), diags: Vec::new(), memo: HashMap::new(), units: 0 }
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
            Scheme::Curried => {
                for (j, v) in own.iter().enumerate() {
                    env2.0.insert(*v, Loc::Hole { level: depth, path: gamma(j as u32 + 1) });
                }
                self.curried(p, &own, atoms, env, env2, depth)
            }
            Scheme::Inst => {
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

    fn diag(&mut self, sev: Severity, code: &'static str, message: String) {
        if !self.diags.iter().any(|d| d.code == code && d.message == message) {
            self.diags.push(Diag { code, severity: sev, span: None, message });
        }
    }

    /// The curried erection (draft 3 §6.7; F1R3Comb v0.6 Req. 5.27, Rem. 5.21):
    /// one `cstar_A[w](t, f)` and one `e(f)` per atom mentioning the unit's
    /// names, the address fanned out to them by a `d`-tree on static
    /// channels. The gates' `b` are static. A gate whose stored body mentions
    /// the unit's names is built by the recipe of Req. 4.9 — its scoped atom
    /// by `cstar`, the closed rest joined by `cons_|`, the store by `cons_q`
    /// — which conforms only when one atom of the body is scoped; otherwise
    /// the join is emitted, reported as `comb-curried-store` (Mat v0.5
    /// Hazard 3.1), and rejected by the discipline check. An atom mentioning
    /// a parent unit's names has no curried template at all
    /// (`comb-curried-reach`).
    fn curried(&mut self, p: &Node, own: &[u32], atoms: Vec<&Node>, env: &Env, mut env2: Env, depth: u32) -> Vec<Atom> {
        let key = p.hash();
        let mut nb = 0;
        for v in own {
            if self.static_vars.contains(v) {
                env2.0.insert(*v, Loc::Static(static_name(&key, 0, nb)));
                nb += 1;
            }
        }
        let mut sc_i = 0u32;
        let mut sc = |role: u32| {
            sc_i += 1;
            static_name(&key, role, sc_i)
        };
        enum Want {
            Erect(Term),
            Feed(Term, CName),
        }
        let mut outside = Vec::new();
        let mut wants = Vec::new();
        let mut gadgets = Vec::new();
        for a in atoms {
            if !mentions(a, own) {
                outside.push(self.atom(a, env, depth));
                continue;
            }
            let at = self.atom(a, &env2, depth + 1);
            if at.free_holes() == 0 {
                outside.push(at);
                continue;
            }
            if at.shape() != Shape::Q {
                if curried_template(&at) {
                    wants.push(Want::Erect(Term::atom(at)));
                } else {
                    self.diag(Severity::Error, "comb-curried-reach", format!(
                        "{} mentions the names of an enclosing unit; the curried template has no parameter for them",
                        at.shape().name()
                    ));
                }
                continue;
            }
            let b = at.names()[0].clone();
            let (scoped, rest): (Vec<Atom>, Vec<Atom>) =
                at.store().unwrap().atoms().iter().cloned().partition(|x| x.free_holes() > 0);
            if !b.is_closed() || !scoped.iter().all(curried_template) {
                self.diag(Severity::Error, "comb-curried-reach", "a stored body mentions the names of an enclosing unit".into());
                continue;
            }
            if scoped.len() > 1 {
                self.diag(Severity::Warning, "comb-curried-store", format!(
                    "a gate's stored body mentions the unit's names in {} atoms; the curried erection must join them \
                     with cons_| at static channels (Mat v0.5 Hazard 3.1)",
                    scoped.len()
                ));
            }
            let mut chans = Vec::new();
            for x in scoped {
                let g = sc(2);
                wants.push(Want::Feed(Term::atom(x), g.clone()));
                chans.push(g);
            }
            if !rest.is_empty() {
                let k = sc(3);
                gadgets.push(Atom::m(k.clone(), CName::quote(Term::from_atoms(rest))));
                chans.push(k);
            }
            let mut acc = chans[0].clone();
            for nxt in chans.into_iter().skip(1) {
                let o = sc(3);
                gadgets.push(Atom::cons(acc, nxt, o.clone()));
                acc = o;
            }
            let (kb, f) = (sc(3), sc(3));
            gadgets.push(Atom::m(kb.clone(), b));
            gadgets.push(Atom::cons_member(Shape::Q, vec![kb, acc, f.clone()]).unwrap());
            gadgets.push(Atom::e(f));
        }
        // the address, once per erecting constructor (Req. 5.12)
        let n = wants.len();
        let mut frontier = vec![r_star()];
        while frontier.len() < n {
            let ch = frontier.remove(0);
            let (x, y) = (sc(1), sc(1));
            outside.push(Atom::d(ch, x.clone(), y.clone()));
            frontier.push(x);
            frontier.push(y);
        }
        for (t, w) in frontier.into_iter().zip(wants) {
            match w {
                Want::Erect(tm) => {
                    let f = sc(4);
                    outside.push(Atom::cstar(t, f.clone(), tm));
                    outside.push(Atom::e(f));
                }
                Want::Feed(tm, g) => outside.push(Atom::cstar(t, g, tm)),
            }
        }
        outside.extend(gadgets);
        outside
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

/// A static channel of the curried erection: a function of the unit's
/// phase-one image, the channel's role and its index (Req. 6.4). Its top
/// node is `s`, so it is neither a leaf, a root, nor a translated name.
pub fn static_name(key: &Hash32, role: u32, i: u32) -> CName {
    let kchain = |n: u32| {
        let mut c = CName::nil();
        for _ in 0..n {
            c = CName::quote(Term::atom(Atom::k(c)));
        }
        c
    };
    let mut h = CName::nil();
    for b in 0..64 {
        let bit = (key.0[b / 8] >> (7 - b % 8)) & 1 == 1;
        h = CName::quote(Term::atom(if bit { Atom::k(h) } else { Atom::br(h, CName::nil()) }));
    }
    CName::quote(Term::atom(Atom::s(h, kchain(role), kchain(i))))
}

/// The curried validity class (Mat v0.5 Req. 3.2): one atom whose name
/// arguments are paths beneath the innermost hole or closed names, with no
/// store or context mentioning a hole.
pub fn curried_template(a: &Atom) -> bool {
    a.store().is_none_or(|s| s.is_closed())
        && a.context().is_none()
        && a.names().iter().all(|n| n.is_closed() || f1r3comb_term::addr::spine(n).0.is_hole() == Some(0))
}

/// A leaf beneath a hole or a root, or a positional name: never an encoding
/// of a source name (Lem. 7.6).
pub fn is_apparatus(n: &CName) -> bool {
    f1r3comb_term::addr::is_address(n) || *n == r_star() || *n == f_star() || {
        let (b, _) = f1r3comb_term::addr::spine(n);
        b.is_hole().is_some()
    }
}
