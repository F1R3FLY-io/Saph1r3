use super::*;
use f1r3comb_term::{Atom, CName};

fn seed_one(c: &CName) -> CName {
    CName::quote(Term::atom(Atom::k(c.clone())))
}

/// `n` distinct names by iterated quotation (experiments.py `fresh_names`).
pub(crate) fn fresh(cur: &mut CName, n: usize) -> Vec<CName> {
    (0..n)
        .map(|_| {
            *cur = seed_one(cur);
            cur.clone()
        })
        .collect()
}

pub(crate) fn fan(k: usize, d: usize) -> Term {
    let mut c = CName::nil();
    let names = fresh(&mut c, k * (d + 1) + 1);
    let v = names.last().unwrap().clone();
    let mut atoms = Vec::new();
    for i in 0..k {
        let b = i * (d + 1);
        for j in 0..d {
            atoms.push(Atom::fw(names[b + j].clone(), names[b + j + 1].clone()));
        }
        atoms.push(Atom::m(names[b].clone(), v.clone()));
    }
    Term::from_atoms(atoms)
}

pub(crate) fn contended(c: usize) -> Term {
    let mut cur = CName::nil();
    let n = fresh(&mut cur, 3);
    let mut atoms = Vec::new();
    for _ in 0..c {
        atoms.push(Atom::fw(n[0].clone(), n[1].clone()));
        atoms.push(Atom::m(n[0].clone(), n[2].clone()));
    }
    Term::from_atoms(atoms)
}

pub(crate) fn dup_tree(h: u32) -> Term {
    let mut cur = CName::nil();
    let names = fresh(&mut cur, 1 << (h + 2));
    let mut atoms = Vec::new();
    let mut work = vec![(0usize, 0u32)];
    while let Some((node, depth)) = work.pop() {
        if depth == h {
            atoms.push(Atom::k(names[node].clone()));
            continue;
        }
        let (l, r) = (2 * node + 1, 2 * node + 2);
        atoms.push(Atom::d(names[node].clone(), names[l].clone(), names[r].clone()));
        work.push((l, depth + 1));
        work.push((r, depth + 1));
    }
    atoms.push(Atom::m(names[0].clone(), names.last().unwrap().clone()));
    Term::from_atoms(atoms)
}

pub(crate) fn construct_eval(k: usize) -> Term {
    let mut cur = CName::nil();
    let mut atoms = Vec::new();
    for _ in 0..k {
        let n = fresh(&mut cur, 5);
        atoms.push(Atom::cons_m(n[0].clone(), n[1].clone(), n[2].clone()));
        atoms.push(Atom::m(n[0].clone(), n[3].clone()));
        atoms.push(Atom::m(n[1].clone(), n[4].clone()));
        atoms.push(Atom::e(n[2].clone()));
    }
    Term::from_atoms(atoms)
}

fn run_host(t: &Term, seed: u64, resolver: Resolver) -> RunReport {
    let cfg = Config { seed, resolver, ..Default::default() };
    run(t, &cfg, &mut Host)
}

fn width(r: &RunReport) -> u64 {
    r.steps.iter().map(|s| s.fired).max().unwrap_or(0)
}

fn max_enum(r: &RunReport) -> u64 {
    r.steps.iter().map(|s| s.enumerated).max().unwrap_or(0)
}

/// The laws of results.txt, E1–E4, over five seeds.
#[test]
fn experiment_laws() {
    for seed in 0..5 {
        for k in [1usize, 4, 16, 64] {
            let r = run_host(&fan(k, 8), seed, Resolver::MaximalProgress);
            assert_eq!((r.steps.len() as u64, r.fired, width(&r), max_enum(&r)), (8, 8 * k as u64, k as u64, k as u64));
            assert_eq!(r.stop, Stop::Quiescent);
            let q = run_host(&fan(k, 8), seed, Resolver::SingleUniform);
            assert_eq!(q.steps.len() as u64, 8 * k as u64);
        }
        for c in [2usize, 4, 8, 16] {
            let r = run_host(&contended(c), seed, Resolver::MaximalProgress);
            assert_eq!((r.steps.len() as u64, r.fired, width(&r), max_enum(&r)), (1, c as u64, c as u64, (c * c) as u64));
        }
        for h in [2u32, 4, 6] {
            let r = run_host(&dup_tree(h), seed, Resolver::MaximalProgress);
            let fired = (1u64 << (h + 1)) - 1;
            assert_eq!((r.steps.len() as u64, r.fired, width(&r)), (h as u64 + 1, fired, 1 << h));
            assert!(r.final_term.is_nil());
        }
        for k in [1usize, 8, 32] {
            let r = run_host(&construct_eval(k), seed, Resolver::MaximalProgress);
            assert_eq!((r.steps.len() as u64, r.fired, width(&r)), (2, 2 * k as u64, k as u64));
            // each gadget leaves m(u, v) behind
            assert_eq!(r.final_term.len(), k);
            assert!(r.final_term.atoms().iter().all(|a| a.shape() == Shape::M));
        }
    }
}

struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = mix64(self.0.wrapping_add(0x9e3779b97f4a7c15));
        self.0
    }
    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
}

fn random_term(g: &mut Lcg) -> Term {
    let mut cur = CName::nil();
    let pool = fresh(&mut cur, 4);
    let pick = |g: &mut Lcg| pool[g.below(pool.len() as u64) as usize].clone();
    let n = 2 + g.below(12) as usize;
    let mut atoms = Vec::new();
    for _ in 0..n {
        let s = Shape::ALL[g.below(13) as usize];
        let a = if s == Shape::Q {
            Atom::q(pick(g), Term::atom(Atom::m(pick(g), pick(g))))
        } else {
            let names = (0..s.arity()).map(|_| pick(g)).collect();
            Atom::new(s, names).unwrap()
        };
        atoms.push(a);
        // bias towards messages, which every rule consumes
        if g.below(2) == 0 {
            atoms.push(Atom::m(pick(g), pick(g)));
        }
    }
    Term::from_atoms(atoms)
}

#[test]
fn finders_agree_and_rounds_equal_greedy() {
    let mut g = Lcg(17);
    let mut nonempty = 0;
    for trial in 0..400u64 {
        let t = random_term(&mut g);
        let mut it = InternTable::default();
        let st = State::import(&t, &mut it).unwrap();
        let a = find_naive(&st);
        let b = find_bucketed(&st);
        assert_eq!(a, b, "trial {trial}");
        if a.is_empty() {
            continue;
        }
        nonempty += 1;
        let greedy = select_greedy(&st, &a, &priority_order(&a, trial, 3));
        let (rounds, _) = select_rounds(&st, &a, trial, 3);
        assert_eq!(greedy, rounds, "trial {trial}");
        // independence and maximality
        let mut used = std::collections::HashSet::new();
        for i in &greedy {
            for r in row_refs(&a[*i]) {
                assert!(used.insert(r));
            }
        }
        for r in &a {
            assert!(row_refs(r).any(|x| used.contains(&x)), "not maximal");
        }
    }
    assert!(nonempty > 200);
}

#[test]
fn symmetric_matches_are_one_redex() {
    let mut cur = CName::nil();
    let n = fresh(&mut cur, 3);
    let t = Term::from_atoms(vec![
        Atom::cons(n[0].clone(), n[0].clone(), n[1].clone()),
        Atom::m(n[0].clone(), n[2].clone()),
        Atom::m(n[0].clone(), n[2].clone()),
    ]);
    let mut it = InternTable::default();
    let st = State::import(&t, &mut it).unwrap();
    assert_eq!(find_bucketed(&st).len(), 1);
}

#[test]
fn deterministic_and_seed_sensitive_only_in_choice() {
    let t = contended(8);
    let a = run_host(&t, 7, Resolver::MaximalProgress);
    let b = run_host(&t, 7, Resolver::MaximalProgress);
    assert_eq!(a.steps, b.steps);
    assert_eq!(a.final_hash(), b.final_hash());
}

#[test]
fn eval_and_quote_rules() {
    // fw(a,b) | q(a, P) -> m(b, @P);  e(b) | m(b, @P) -> P
    let mut cur = CName::nil();
    let n = fresh(&mut cur, 3);
    let p = Term::atom(Atom::m(n[2].clone(), CName::nil()));
    let t = Term::from_atoms(vec![
        Atom::fw(n[0].clone(), n[1].clone()),
        Atom::q(n[0].clone(), p.clone()),
        Atom::e(n[1].clone()),
    ]);
    let r = run_host(&t, 0, Resolver::MaximalProgress);
    assert_eq!(r.final_term, p);
    assert_eq!(r.steps.len(), 2);
}
