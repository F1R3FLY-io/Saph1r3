//! The phase-two experiments of `phase2.py` (E5–E7), three arms, on this
//! machine (Mat v0.5 Obl. 11.2; F1R3Comb v0.6 Obl. 11.3). The images are
//! hand-built exactly as `phase2.py` builds them; the literal arm is the
//! negative control no product crate can emit.

use f1r3comb_mat::{InternTable, State};
use f1r3comb_par::{Config, Host, Resolver, StepEngine};
use f1r3comb_term::addr::{hole_leaf, k_const, spine};
use f1r3comb_term::rules::rule_index;
use f1r3comb_term::{Atom, CName, Shape, Term};

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Arm {
    Literal,
    Curried,
    Inst,
}

/// `experiments.py`'s `fresh_names`: iterated `k`-quotation from a cursor.
pub struct Fresh(CName);
impl Fresh {
    pub fn new() -> Fresh {
        Fresh(CName::nil())
    }
    pub fn next(&mut self) -> CName {
        self.0 = CName::quote(Term::atom(Atom::k(self.0.clone())));
        self.0.clone()
    }
    pub fn take(&mut self, n: usize) -> Vec<CName> {
        (0..n).map(|_| self.next()).collect()
    }
}

impl Default for Fresh {
    fn default() -> Self {
        Fresh::new()
    }
}

fn owner(n: &CName) -> Option<CName> {
    let (b, w) = spine(n);
    if w.is_empty() {
        None
    } else {
        Some(b)
    }
}

const PATHS: [[bool; 3]; 8] = [
    [false, false, false],
    [false, false, true],
    [false, true, false],
    [false, true, true],
    [true, false, false],
    [true, false, true],
    [true, true, false],
    [true, true, true],
];

fn run_to_quiescence(t: &Term, seed: u64, parallel: bool) -> (u64, Term) {
    let cfg = Config {
        seed,
        resolver: if parallel { Resolver::MaximalProgress } else { Resolver::SingleUniform },
        ..Default::default()
    };
    let r = f1r3comb_par::run(t, &cfg, &mut Host);
    (r.steps.len() as u64, r.final_term)
}

fn e5_image(s: &[CName], arm: Arm) -> Vec<Atom> {
    let l = hole_leaf(0, &[false]);
    let r = hole_leaf(0, &[true]);
    match arm {
        Arm::Literal => vec![
            Atom::d(s[0].clone(), s[1].clone(), s[2].clone()),
            Atom::cons_m(s[1].clone(), s[3].clone(), s[5].clone()),
            Atom::m(s[3].clone(), CName::nil()),
            Atom::cons_m(s[2].clone(), s[4].clone(), s[6].clone()),
            Atom::m(s[4].clone(), k_const()),
            Atom::d(s[5].clone(), s[7].clone(), s[8].clone()),
            Atom::cons_member(Shape::Fw, vec![s[7].clone(), s[6].clone(), s[9].clone()]).unwrap(),
            Atom::e(s[9].clone()),
            Atom::cons_member(Shape::K, vec![s[8].clone(), s[10].clone()]).unwrap(),
            Atom::e(s[10].clone()),
        ],
        Arm::Curried => vec![
            Atom::d(s[0].clone(), s[1].clone(), s[2].clone()),
            Atom::cstar(s[1].clone(), s[9].clone(), Term::atom(Atom::fw(l.clone(), r))),
            Atom::e(s[9].clone()),
            Atom::cstar(s[2].clone(), s[10].clone(), Term::atom(Atom::k(l))),
            Atom::e(s[10].clone()),
        ],
        Arm::Inst => vec![
            Atom::inst(s[0].clone(), s[9].clone(), Term::from_atoms(vec![Atom::fw(l.clone(), r), Atom::k(l)])),
            Atom::e(s[9].clone()),
        ],
    }
}

/// E5: a unit of two top-level atoms, `n` concurrent instances. Returns
/// (steps, erected fw, mixed fw).
pub fn e5(arm: Arm, n: usize, seed: u64, parallel: bool) -> (u64, usize, usize) {
    let mut f = Fresh::new();
    let s = f.take(64);
    let img = e5_image(&s, arm);
    let mut atoms = Vec::new();
    for a in f.take(n) {
        atoms.extend(img.iter().cloned());
        atoms.push(Atom::m(s[0].clone(), a));
    }
    let (steps, fin) = run_to_quiescence(&Term::from_atoms(atoms), seed, parallel);
    let fws: Vec<&Atom> = fin.atoms().iter().filter(|a| a.shape() == Shape::Fw).collect();
    assert_eq!(fws.len(), n, "{arm:?}: every instance erects its forwarder");
    let mixed = fws.iter().filter(|a| owner(&a.names()[0]) != owner(&a.names()[1])).count();
    (steps, fws.len(), mixed)
}

fn dx_image(arm: Arm, x: &CName, r: &CName, s: &[CName]) -> Vec<Atom> {
    let lv: Vec<CName> = PATHS.iter().map(|p| hole_leaf(0, p)).collect();
    let (p0, q1, p1, p2, c, b, c2, nxt) =
        (&lv[0], &lv[1], &lv[2], &lv[3], &lv[4], &lv[5], &lv[6], &lv[7]);
    match arm {
        Arm::Inst => {
            let ctx = Term::from_atoms(vec![
                Atom::d(x.clone(), p0.clone(), q1.clone()),
                Atom::d(q1.clone(), p1.clone(), p2.clone()),
                Atom::s(p0.clone(), b.clone(), c2.clone()),
                Atom::q(b.clone(), Term::atom(Atom::e(c.clone()))),
                Atom::e(c2.clone()),
                Atom::fw(p1.clone(), x.clone()),
                Atom::fw(p2.clone(), c.clone()),
                Atom::m(r.clone(), nxt.clone()),
            ]);
            vec![Atom::inst(r.clone(), s[0].clone(), ctx), Atom::e(s[0].clone())]
        }
        Arm::Curried => {
            let bs = s[63].clone();
            let tmpl = [
                Atom::d(x.clone(), p0.clone(), q1.clone()),
                Atom::d(q1.clone(), p1.clone(), p2.clone()),
                Atom::s(p0.clone(), bs.clone(), c2.clone()),
                Atom::e(c2.clone()),
                Atom::fw(p1.clone(), x.clone()),
                Atom::fw(p2.clone(), c.clone()),
                Atom::m(r.clone(), nxt.clone()),
            ];
            let targets = tmpl.len() + 1;
            // the d-tree on static channels, breadth first, as phase2.py
            let mut atoms = Vec::new();
            let mut frontier = vec![r.clone()];
            let mut k = 0;
            while frontier.len() < targets {
                let ch = frontier.remove(0);
                let (a, bb) = (s[k].clone(), s[k + 1].clone());
                k += 2;
                atoms.push(Atom::d(ch, a.clone(), bb.clone()));
                frontier.push(a);
                frontier.push(bb);
            }
            let outs = &frontier[..targets];
            let mut k = 40;
            for (t, a) in outs.iter().zip(tmpl.iter()) {
                let f = s[k].clone();
                k += 1;
                atoms.push(Atom::cstar(t.clone(), f.clone(), Term::atom(a.clone())));
                atoms.push(Atom::e(f));
            }
            let (g, kb, f) = (s[k].clone(), s[k + 1].clone(), s[k + 2].clone());
            atoms.push(Atom::cstar(outs[targets - 1].clone(), g.clone(), Term::atom(Atom::e(c.clone()))));
            atoms.push(Atom::m(kb.clone(), bs));
            atoms.push(Atom::cons_member(Shape::Q, vec![kb, g, f.clone()]).unwrap());
            atoms.push(Atom::e(f));
            atoms
        }
        Arm::Literal => unreachable!("E6 has no literal arm"),
    }
}

/// E6: the recursion combinator of draft 3 Ex. 6.9, `n` unfoldings. Returns
/// the per-unfolding step counts and names built.
pub fn e6(arm: Arm, n: usize, seed: u64, parallel: bool) -> (Vec<u64>, Vec<u64>) {
    let mut fr = Fresh::new();
    let xr = fr.take(2);
    let (x, r) = (&xr[0], &xr[1]);
    let s = fr.take(64);
    let img = dx_image(arm, x, r, &s);
    let v = CName::quote(Term::from_atoms(img.clone()));
    let sigma = fr.next();
    let mut atoms = img;
    atoms.push(Atom::m(x.clone(), v));
    atoms.push(Atom::m(r.clone(), sigma));
    let cfg = Config {
        seed,
        resolver: if parallel { Resolver::MaximalProgress } else { Resolver::SingleUniform },
        ..Default::default()
    };
    let trigger = match arm {
        Arm::Inst => rule_index("inst").unwrap(),
        _ => rule_index("dist").unwrap(),
    };
    let tshape = if arm == Arm::Inst { Shape::Inst } else { Shape::D };
    let mut it = InternTable::default();
    let mut st = State::import(&Term::from_atoms(atoms), &mut it).unwrap();
    let r_id = it.intern(r).unwrap();
    let mut marks: Vec<(u64, u64)> = Vec::new();
    let mut names = 0u64;
    let mut step = 0u64;
    while marks.len() < n && step < 100_000 {
        let (rs, chosen) = Host.find_and_select(&st, &cfg, step);
        if chosen.is_empty() {
            break;
        }
        for i in &chosen {
            let rd = &rs[*i];
            if rd.rule as usize == trigger && st.table(tshape).cols[0][rd.consumer as usize] == r_id {
                marks.push((step, names));
            }
        }
        f1r3comb_par::commit_counting(&mut st, &mut it, &rs, &chosen, &mut names).unwrap();
        step += 1;
    }
    let ds = marks.windows(2).map(|w| w[1].0 - w[0].0).collect();
    let dn = marks.windows(2).map(|w| w[1].1 - w[0].1).collect();
    (ds, dn)
}

/// E7: a store whose payload mentions scoped names in two atoms. Returns
/// the number of mixed stores.
pub fn e7(arm: Arm, n: usize, seed: u64) -> usize {
    let mut f = Fresh::new();
    let s = f.take(64);
    let (c1, c2, c3) = (hole_leaf(0, &PATHS[0]), hole_leaf(0, &PATHS[1]), hole_leaf(0, &PATHS[2]));
    let b = s[63].clone();
    let img = match arm {
        Arm::Inst => {
            let payload = Term::from_atoms(vec![Atom::fw(c1, c3), Atom::k(c2)]);
            vec![Atom::inst(s[0].clone(), s[9].clone(), Term::atom(Atom::q(b, payload))), Atom::e(s[9].clone())]
        }
        Arm::Curried => vec![
            Atom::d(s[0].clone(), s[1].clone(), s[2].clone()),
            Atom::cstar(s[1].clone(), s[3].clone(), Term::atom(Atom::fw(c1, c3))),
            Atom::cstar(s[2].clone(), s[4].clone(), Term::atom(Atom::k(c2))),
            Atom::cons(s[3].clone(), s[4].clone(), s[5].clone()),
            Atom::m(s[6].clone(), b),
            Atom::cons_member(Shape::Q, vec![s[6].clone(), s[5].clone(), s[7].clone()]).unwrap(),
            Atom::e(s[7].clone()),
        ],
        Arm::Literal => unreachable!(),
    };
    let mut atoms = Vec::new();
    for a in f.take(n) {
        atoms.extend(img.iter().cloned());
        atoms.push(Atom::m(s[0].clone(), a));
    }
    let (_, fin) = run_to_quiescence(&Term::from_atoms(atoms), seed, true);
    let qs: Vec<&Atom> = fin.atoms().iter().filter(|a| a.shape() == Shape::Q).collect();
    assert_eq!(qs.len(), n);
    qs.iter()
        .filter(|q| {
            let mut owners = std::collections::HashSet::new();
            for at in q.store().unwrap().atoms() {
                for x in at.names() {
                    owners.insert(owner(x).map(|o| o.encode().to_vec()));
                }
            }
            owners.len() > 1
        })
        .count()
}

/// The recursion combinator as *this compiler* emits it:
/// `D_x | x!(D_x)` with `D_x = for(y <- x){ x!(*y) | *y }`. Returns the
/// per-unfolding step counts and names built, an unfolding being marked by
/// each `inst` firing at `r*`.
pub fn compiled_dx(n: usize, seed: u64, parallel: bool) -> (Vec<u64>, Vec<u64>) {
    let x = crate::B;
    let src = format!("for(y <- {x}){{ {x}!(*y) | *y }} | {x}!(for(y <- {x}){{ {x}!(*y) | *y }})");
    let c = crate::compile(&src, f1r3comb_term::artefact::Scheme::Inst);
    let (t, _) = f1r3comb_par::deploy(&c.term, &mut f1r3comb_term::addr::Allocator::new());
    let cfg = Config {
        seed,
        resolver: if parallel { Resolver::MaximalProgress } else { Resolver::SingleUniform },
        ..Default::default()
    };
    let inst = rule_index("inst").unwrap();
    let mut it = InternTable::default();
    let mut st = State::import(&t, &mut it).unwrap();
    let (mut marks, mut names, mut step) = (Vec::new(), 0u64, 0u64);
    while marks.len() < n && step < 200_000 {
        let (rs, chosen) = Host.find_and_select(&st, &cfg, step);
        if chosen.is_empty() {
            break;
        }
        for i in &chosen {
            if rs[*i].rule as usize == inst {
                marks.push((step, names));
            }
        }
        f1r3comb_par::commit_counting(&mut st, &mut it, &rs, &chosen, &mut names).unwrap();
        step += 1;
    }
    (marks.windows(2).map(|w| w[1].0 - w[0].0).collect(), marks.windows(2).map(|w| w[1].1 - w[0].1).collect())
}
