use super::*;
use f1r3comb_term::addr::hole_leaf;

fn seed_one(c: &CName) -> CName {
    CName::quote(Term::atom(Atom::k(c.clone())))
}
fn seed_zero(c: &CName) -> CName {
    CName::quote(Term::atom(Atom::bl(c.clone(), CName::nil())))
}

fn sample() -> Term {
    let z = CName::nil();
    let a = seed_one(&z);
    let b = seed_zero(&a);
    let inner = Term::from_atoms(vec![Atom::m(a.clone(), b.clone()), Atom::e(b.clone())]);
    Term::from_atoms(vec![
        Atom::m(z.clone(), CName::quote(inner.clone())),
        Atom::d(z.clone(), a.clone(), b.clone()),
        Atom::q(a.clone(), inner),
        Atom::m(z.clone(), z.clone()),
        Atom::m(z.clone(), z.clone()),
        Atom::cons_member(Shape::S, vec![a.clone(), b.clone(), z.clone(), a.clone()]).unwrap(),
        Atom::inst(a.clone(), b.clone(), Term::from_atoms(vec![Atom::fw(hole_leaf(0, &[true]), a.clone()), Atom::inst(z.clone(), a, Term::atom(Atom::m(CName::hole(0), CName::hole(1))))])),
    ])
}

#[test]
fn import_export_identity() {
    let t = sample();
    let mut it = InternTable::default();
    let st = State::import(&t, &mut it).unwrap();
    assert_eq!(st.rows(), t.len() as u64);
    assert_eq!(st.export(&mut it).encode(), t.encode());
    // export then import is the identity on markings
    let st2 = State::import(&st.export(&mut it), &mut it).unwrap();
    let mut a = st.all_rows();
    let mut b = st2.all_rows();
    a.sort();
    b.sort();
    assert_eq!(a, b);
}

#[test]
fn key_and_hash_agree() {
    let t = sample();
    let mut it = InternTable::default();
    State::import(&t, &mut it).unwrap();
    let mut seen = HashMap::new();
    for i in 0..it.len() {
        let h = it.chash(NameId(i as u32));
        assert!(seen.insert(h, i).is_none(), "two ids, one hash");
    }
}

#[test]
fn cmat_round_trip_and_canonical() {
    let t = sample();
    let mut it1 = InternTable::default();
    let st1 = State::import(&t, &mut it1).unwrap();
    let b1 = cmat_encode(&st1, &mut it1);
    // a different interning history must not change the file
    let mut it2 = InternTable::default();
    it2.intern(&seed_zero(&seed_zero(&CName::nil()))).unwrap();
    let st2 = State::import(&t, &mut it2).unwrap();
    assert_eq!(cmat_encode(&st2, &mut it2), b1);
    let (st3, mut it3) = cmat_decode(&b1, NameBudget::default()).unwrap();
    assert_eq!(st3.export(&mut it3).encode(), t.encode());
    let mut bad = b1.clone();
    bad[5] = b'B';
    assert_eq!(cmat_decode(&bad, NameBudget::default()).err(), Some(CmatError::Presentation(b'B')));
}

#[test]
fn budget_halts() {
    let mut it = InternTable::new(NameBudget { max_names: 2, max_components: 8, max_arena: 64 });
    let t = sample();
    assert!(State::import(&t, &mut it).is_err());
}

#[test]
fn fill_agrees_with_the_term_level_fill() {
    let t = sample();
    let mut it = InternTable::default();
    State::import(&t, &mut it).unwrap();
    let inst = t.atoms().iter().find(|a| a.shape() == Shape::Inst).unwrap().clone();
    let ctx = inst.context().unwrap().clone();
    let v = seed_one(&seed_one(&seed_one(&CName::nil())));
    let tid = it.intern(&CName::quote(ctx.clone())).unwrap();
    let vid = it.intern(&v).unwrap();
    let (r, built) = it.fill(tid, vid).unwrap();
    assert!(built > 0);
    assert_eq!(it.name(r), CName::quote(f1r3comb_term::fill(&ctx, &v)));
    assert_eq!(it.free(r), 0);
}
