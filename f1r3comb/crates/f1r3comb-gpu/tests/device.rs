//! Device conformance: the GPU's redex set, priorities and step equal the
//! host's, step for step (Mat Req. 6.5). Skipped (with a message) when no
//! adapter is available.
#![cfg(feature = "device")]

use f1r3comb_gpu::device::Gpu;
use f1r3comb_mat::{InternTable, State};
use f1r3comb_par::*;
use f1r3comb_term::addr::hole_leaf;
use f1r3comb_term::{Atom, CName, Shape, Term};

fn seed_one(c: &CName) -> CName {
    CName::quote(Term::atom(Atom::k(c.clone())))
}

fn gpu() -> Option<Gpu> {
    match Gpu::new() {
        Ok(g) => {
            eprintln!("device: {} ({})", g.adapter_name, g.backend);
            Some(g)
        }
        Err(e) => {
            eprintln!("skipping device conformance: {e}");
            None
        }
    }
}

fn fresh(cur: &mut CName, n: usize) -> Vec<CName> {
    (0..n).map(|_| { *cur = seed_one(cur); cur.clone() }).collect()
}

fn lockstep(g: &mut Gpu, t: &Term, seed: u64) -> u64 {
    let mut it = InternTable::default();
    let mut st = State::import(t, &mut it).unwrap();
    let cfg = Config { seed, ..Default::default() };
    let mut fired = 0;
    for step in 0..10_000u64 {
        let (hr, hc) = Host.find_and_select(&st, &cfg, step);
        let (gr, gc) = g.find_and_select(&st, &cfg, step);
        assert_eq!(hr, gr, "redex sets differ at step {step}");
        assert_eq!(hc, gc, "steps differ at step {step}");
        if hc.is_empty() {
            return fired;
        }
        let hp: Vec<u64> = hr.iter().map(|r| priority(seed, step, r)).collect();
        assert_eq!(hp, g.priorities(&st, seed, step), "priorities differ at step {step}");
        fired += hc.len() as u64;
        commit(&mut st, &mut it, &hr, &hc).unwrap();
    }
    panic!("no quiescence");
}

fn families() -> Vec<Term> {
    let mut out = Vec::new();
    // E1 fan
    let mut c = CName::nil();
    let n = fresh(&mut c, 16 * 5 + 1);
    let v = n.last().unwrap().clone();
    let mut a = Vec::new();
    for i in 0..16 {
        for j in 0..4 {
            a.push(Atom::fw(n[i * 5 + j].clone(), n[i * 5 + j + 1].clone()));
        }
        a.push(Atom::m(n[i * 5].clone(), v.clone()));
    }
    out.push(Term::from_atoms(a));
    // E2 contended
    let mut c = CName::nil();
    let n = fresh(&mut c, 3);
    let mut a = Vec::new();
    for _ in 0..16 {
        a.push(Atom::fw(n[0].clone(), n[1].clone()));
        a.push(Atom::m(n[0].clone(), n[2].clone()));
    }
    out.push(Term::from_atoms(a));
    // E4 construct+eval, plus every constructor and the quote rule
    let mut c = CName::nil();
    let mut a = Vec::new();
    for _ in 0..8 {
        let n = fresh(&mut c, 6);
        a.push(Atom::cons_m(n[0].clone(), n[1].clone(), n[2].clone()));
        a.push(Atom::m(n[0].clone(), n[3].clone()));
        a.push(Atom::m(n[1].clone(), n[4].clone()));
        a.push(Atom::e(n[2].clone()));
        a.push(Atom::cons(n[5].clone(), n[5].clone(), n[3].clone()));
        a.push(Atom::m(n[5].clone(), n[4].clone()));
        a.push(Atom::m(n[5].clone(), n[4].clone()));
        a.push(Atom::cons_member(Shape::S, vec![n[1].clone(), n[1].clone(), n[1].clone(), n[2].clone()]).unwrap());
        a.push(Atom::cons_member(Shape::Q, vec![n[0].clone(), n[1].clone(), n[3].clone()]).unwrap());
        a.push(Atom::fw(n[4].clone(), n[0].clone()));
        a.push(Atom::q(n[4].clone(), Term::atom(Atom::m(n[0].clone(), n[1].clone()))));
    }
    out.push(Term::from_atoms(a));
    // inst erection: eight instances of a two-atom unit contending at one
    // static channel (E5's inst arm), plus a nested context
    let mut c = CName::nil();
    let s = fresh(&mut c, 3);
    let l = hole_leaf(0, &[false]);
    let r = hole_leaf(0, &[true]);
    let ctx = Term::from_atoms(vec![
        Atom::fw(l.clone(), r.clone()),
        Atom::k(l.clone()),
        Atom::inst(s[2].clone(), s[2].clone(), Term::atom(Atom::m(CName::hole(0), CName::hole(1)))),
    ]);
    let mut a = Vec::new();
    for x in fresh(&mut c, 8) {
        a.push(Atom::inst(s[0].clone(), s[1].clone(), ctx.clone()));
        a.push(Atom::e(s[1].clone()));
        a.push(Atom::m(s[0].clone(), x.clone()));
        a.push(Atom::m(s[2].clone(), seed_one(&x)));
    }
    out.push(Term::from_atoms(a));
    out
}

#[test]
fn device_matches_host() {
    let Some(mut g) = gpu() else { return };
    for t in families() {
        for seed in 0..3 {
            lockstep(&mut g, &t, seed);
        }
    }
    let h = f1r3comb_term::lattice::top_level(&families()[2]);
    assert!(h.present(Shape::ConsS));
    assert!(f1r3comb_term::lattice::top_level(&families()[3]).present(Shape::Inst));
    eprintln!("stats: {:?}", g.stats);
}

// Compiled K0 programs on the device, judged against the K0 reference
// reducer, are in `f1r3comb-check` (feature `gpu`).
