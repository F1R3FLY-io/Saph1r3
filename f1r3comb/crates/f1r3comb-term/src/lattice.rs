//! The atom multiset and the lattice point (F1R3Comb v0.6 §10; draft 3 §9).
//!
//! [`top_level`] is the `|−|` of rhocomb-logic Def. 2.9, the logic's input.
//! [`hereditary`] counts every atom occurrence reachable through stores,
//! contexts and quotations. Under the `inst` erection a phase-two image is,
//! at top level, one `inst` shape per unit (Req. 10.3): the informative point
//! is the phase-one image's, which `f1r3comb-ir` reports.

use crate::{Shape, Term, SHAPES};
use std::collections::HashSet;
use std::fmt::Write;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Counts(pub [u64; SHAPES]);

impl Default for Counts {
    fn default() -> Self {
        Counts([0; SHAPES])
    }
}

impl Counts {
    pub fn get(&self, s: Shape) -> u64 {
        self.0[s.ix()]
    }
    pub fn present(&self, s: Shape) -> bool {
        self.get(s) > 0
    }
    pub fn total(&self) -> u64 {
        self.0.iter().sum()
    }
    pub fn add(&mut self, s: Shape) {
        self.0[s.ix()] += 1;
    }
}

pub fn top_level(t: &Term) -> Counts {
    let mut c = Counts::default();
    for a in t.atoms() {
        c.add(a.shape());
    }
    c
}

pub fn hereditary(t: &Term) -> Counts {
    let mut c = Counts::default();
    let mut seen: HashSet<Vec<u8>> = HashSet::new();
    let mut work = vec![t.clone()];
    while let Some(t) = work.pop() {
        for a in t.atoms() {
            c.add(a.shape());
            if let Some(p) = a.store().or(a.context()) {
                work.push(p.clone());
            }
            for n in a.names() {
                if seen.insert(n.encode().to_vec()) {
                    work.push(n.drop());
                }
            }
        }
    }
    c
}

/// A point of draft 3 Fig. 1's product lattice.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Point {
    pub computation: Vec<Shape>,
    pub construction: Vec<Shape>,
    pub routers: Vec<Shape>,
    pub erection: Vec<Shape>,
}

pub fn point(c: &Counts) -> Point {
    let pick = |f: &dyn Fn(Shape) -> bool| Shape::ALL.iter().copied().filter(|s| f(*s) && c.present(*s)).collect::<Vec<_>>();
    Point {
        computation: pick(&|s| matches!(s, Shape::D | Shape::E)),
        construction: pick(&|s| s.is_constructor()),
        routers: pick(&|s| matches!(s, Shape::K | Shape::Fw | Shape::Bl | Shape::Br | Shape::S | Shape::Q)),
        erection: pick(&|s| matches!(s, Shape::Inst | Shape::Cstar)),
    }
}

fn set(xs: &[Shape]) -> String {
    let v: Vec<&str> = xs.iter().map(|s| s.name()).collect();
    format!("{{{}}}", v.join(", "))
}

/// Is the term in the decidable modal fragment of rhocomb-logic (which
/// assumes `e` absent)? (Req. 10.2)
pub fn decidable_fragment(c: &Counts) -> bool {
    !c.present(Shape::E)
}

pub fn report_counts(title: &str, top: &Counts, her: &Counts) -> String {
    let p = point(her);
    let mut s = String::new();
    let _ = writeln!(s, "{title}");
    let _ = writeln!(s, "  shape      top-level  hereditary");
    for sh in Shape::ALL {
        if top.present(sh) || her.present(sh) {
            let _ = writeln!(s, "  {:<10} {:>9}  {:>10}", sh.name(), top.get(sh), her.get(sh));
        }
    }
    let _ = writeln!(s, "  total      {:>9}  {:>10}", top.total(), her.total());
    let _ = writeln!(s, "  lattice point: computation {}, construction {}, routers {}, erection {}",
        set(&p.computation), set(&p.construction), set(&p.routers), set(&p.erection));
    let _ = writeln!(
        s,
        "  decidable fragment of the logic: {}",
        if decidable_fragment(her) { "yes" } else { "no (contains e)" }
    );
    s
}

pub fn report(t: &Term) -> String {
    report_counts("target image", &top_level(t), &hereditary(t))
}
