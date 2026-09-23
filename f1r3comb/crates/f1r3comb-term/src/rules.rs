//! The rule table of draft 3 Def. 3.3 and 3.8 and §6.7, **generated** from
//! the shape table (F1R3Comb v0.6 Req. 4.5; Mat v0.5 Req. 5.1).
//!
//! Rows, in canonical order: the seven routing and destructor rules
//! (`dist`, `kill`, `forward`, `bind_l`, `bind_r`, `synch`, `eval`), `quote`,
//! `make_|`, one `make_A` per base shape `A` (the whole family; a program
//! uses the subset its header records as `F(P)`), and the two erecting rules
//! `inst` and `cstar`, which share one encode (Mat Decision 2.6). A redex's
//! canonical key begins with its rule's index here.

use crate::{Shape, BASE};
use std::sync::OnceLock;

/// A variable of a conclusion template.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum Var {
    /// The consumer's argument `j`.
    Slot(u8),
    /// The payload (second argument) of premise `j`. For a quiet premise this
    /// is the quotation of the stored process (Mat Decision 3.2).
    Payload(u8),
}

/// Which table a premise joins against.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum Datum {
    /// `m(a, v)`.
    Loud,
    /// `q(a, p)`: readable only by a forwarder.
    Quiet,
}

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub struct Premise {
    /// The consumer argument whose name must equal the premise's subject.
    pub subject: u8,
    pub datum: Datum,
}

#[derive(Clone, PartialEq, Eq, Debug)]
pub enum NodeTemplate {
    /// `*u | *r`, the multiset union of two quoted processes.
    Union(Var, Var),
    /// A single atom over names; for `q` the second name is the quotation of
    /// the store.
    Atom(Shape, Vec<Var>),
    /// `T[v]`: a template (a context) filled with a name.
    Instantiate { template: Var, with: Var },
}

#[derive(Clone, PartialEq, Eq, Debug)]
pub enum Conclusion {
    /// Class (a): a fixed multiset of atoms over premise names.
    Atoms(Vec<(Shape, Vec<Var>)>),
    /// Class (b): `m(deliver, @node)`.
    Encode { deliver: Var, node: NodeTemplate },
    /// Class (c): `*payload`, the process the name quotes.
    Decode { payload: Var },
}

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum RuleClass {
    A,
    B,
    C,
}

#[derive(Clone, Debug)]
pub struct RuleSpec {
    pub name: String,
    pub consumer: Shape,
    pub premises: Vec<Premise>,
    pub conclusion: Conclusion,
    pub class: RuleClass,
}

use Datum::*;
use Var::*;

fn loud(n: usize) -> Vec<Premise> {
    (0..n).map(|j| Premise { subject: j as u8, datum: Loud }).collect()
}

fn class(c: &Conclusion) -> RuleClass {
    match c {
        Conclusion::Atoms(_) => RuleClass::A,
        Conclusion::Encode { .. } => RuleClass::B,
        Conclusion::Decode { .. } => RuleClass::C,
    }
}

fn generate() -> Vec<RuleSpec> {
    let mut v = Vec::new();
    let mut push = |name: &str, consumer: Shape, premises: Vec<Premise>, conclusion: Conclusion| {
        let class = class(&conclusion);
        v.push(RuleSpec { name: name.to_string(), consumer, premises, conclusion, class });
    };
    use Shape as S;
    push("dist", S::D, loud(1), Conclusion::Atoms(vec![(S::M, vec![Slot(1), Payload(0)]), (S::M, vec![Slot(2), Payload(0)])]));
    push("kill", S::K, loud(1), Conclusion::Atoms(vec![]));
    push("forward", S::Fw, loud(1), Conclusion::Atoms(vec![(S::M, vec![Slot(1), Payload(0)])]));
    push("bind_l", S::Bl, loud(1), Conclusion::Atoms(vec![(S::Fw, vec![Payload(0), Slot(1)])]));
    push("bind_r", S::Br, loud(1), Conclusion::Atoms(vec![(S::Fw, vec![Slot(1), Payload(0)])]));
    push("synch", S::S, loud(1), Conclusion::Atoms(vec![(S::Fw, vec![Slot(1), Slot(2)])]));
    push("eval", S::E, loud(1), Conclusion::Decode { payload: Payload(0) });
    push(
        "quote",
        S::Fw,
        vec![Premise { subject: 0, datum: Quiet }],
        Conclusion::Atoms(vec![(S::M, vec![Slot(1), Payload(0)])]),
    );
    push(
        "make_par",
        S::ConsPar,
        loud(2),
        Conclusion::Encode { deliver: Slot(2), node: NodeTemplate::Union(Payload(0), Payload(1)) },
    );
    for a in Shape::BASES {
        let n = a.arity();
        let c = a.cons_of().unwrap();
        push(
            &format!("make_{}", a.name()),
            c,
            loud(n),
            Conclusion::Encode {
                deliver: Slot(n as u8),
                node: NodeTemplate::Atom(a, (0..n as u8).map(Payload).collect()),
            },
        );
    }
    for (name, s) in [("inst", S::Inst), ("cstar", S::Cstar)] {
        push(
            name,
            s,
            loud(1),
            Conclusion::Encode {
                deliver: Slot(1),
                node: NodeTemplate::Instantiate { template: Slot(2), with: Payload(0) },
            },
        );
    }
    v
}

/// The generated table.
pub fn rules() -> &'static [RuleSpec] {
    static T: OnceLock<Vec<RuleSpec>> = OnceLock::new();
    T.get_or_init(generate)
}

/// 7 routing/destructor + quote + make_| + one make per base shape + 2.
pub const RULE_COUNT: usize = 9 + BASE + 2;

/// Rule indices whose consumer is `shape`, in table order.
pub fn rules_for(shape: Shape) -> impl Iterator<Item = usize> {
    rules().iter().enumerate().filter(move |(_, r)| r.consumer == shape).map(|(i, _)| i)
}

/// Index of the rule named `name`.
pub fn rule_index(name: &str) -> Option<usize> {
    rules().iter().position(|r| r.name == name)
}

/// The grading `Wt` (Mat Erratum 9.1, derived from the table): `m` weighs
/// 0; a shape that consumes weighs one more than the heaviest non-message
/// atom any of its class-(a) rules produces; everything else 1 (so the
/// family and the erecting rules weigh one, Mat v0.5 Req. 5.4).
pub fn weights() -> [u32; crate::SHAPES] {
    let mut w = [1u32; crate::SHAPES];
    w[Shape::M.ix()] = 0;
    for _ in 0..crate::SHAPES {
        for r in rules() {
            if let Conclusion::Atoms(atoms) = &r.conclusion {
                let heaviest = atoms.iter().filter(|(s, _)| *s != Shape::M).map(|(s, _)| w[s.ix()]).max().unwrap_or(0);
                let c = r.consumer.ix();
                w[c] = w[c].max(1 + heaviest);
            }
        }
    }
    w
}
