use super::*;
use crate::addr::*;
use crate::rules::*;

fn nm(i: u32) -> CName {
    let mut c = CName::nil();
    for _ in 0..=i {
        c = CName::quote(Term::atom(Atom::k(c)));
    }
    c
}

#[test]
fn par_is_a_sorted_multiset() {
    let a = Term::atom(Atom::m(nm(1), CName::nil()));
    let b = Term::atom(Atom::fw(nm(2), nm(3)));
    let x = Term::par(vec![a.clone(), b.clone(), Term::nil()]);
    let y = Term::par(vec![b, Term::nil(), a]);
    assert_eq!(x.encode(), y.encode());
    assert_eq!(x.len(), 2);
}

#[test]
fn tag_layout_is_structured() {
    assert_eq!(Shape::M.tag(), 0x32);
    assert_eq!(Shape::Q.tag(), 0x3A);
    assert_eq!(Shape::ConsPar.tag(), 0x51);
    assert_eq!(Shape::ConsM.tag(), 0x52);
    assert_eq!(Shape::ConsQ.tag(), 0x5A);
    assert_eq!(Shape::Inst.tag(), 0x60);
    for s in Shape::ALL {
        assert!(s.tag() > 0x20);
        assert_eq!(Shape::from_tag(s.tag()), Some(s));
        if let Some(b) = s.builds() {
            assert_eq!(s.tag(), 0x50 + (b.tag() - 0x30));
            assert_eq!(s.arity(), b.arity() + 1);
        }
    }
    assert!(matches!(Term::decode(&[0x00]), Err(DecodeError::BadTag(0x00, 0))));
    assert!(matches!(Term::decode(&[tags::DROP, tags::QUOTE, tags::NIL]), Err(DecodeError::BadTag(tags::DROP, 0))));
}

#[test]
fn every_shape_round_trips() {
    let ctx = Term::from_atoms(vec![Atom::fw(hole_leaf(0, &[false]), hole_leaf(0, &[true])), Atom::k(hole_leaf(0, &[false]))]);
    let mut atoms = vec![
        Atom::q(nm(2), Term::atom(Atom::m(nm(1), CName::nil()))),
        Atom::cons(nm(1), nm(2), nm(3)),
        Atom::inst(nm(1), nm(2), ctx.clone()),
        Atom::cstar(nm(1), nm(2), Term::atom(Atom::k(hole_leaf(0, &[true])))),
    ];
    for s in Shape::ALL {
        if s.name_arity() == s.arity() {
            atoms.push(Atom::new(s, (0..s.arity() as u32).map(nm).collect()).unwrap());
        }
    }
    let t = Term::from_atoms(atoms);
    assert!(t.is_closed());
    let back = Term::decode(t.encode()).unwrap();
    assert_eq!(back, t);
    let a = artefact::Artefact {
        header: artefact::Header {
            scheme: artefact::Scheme::Inst,
            source_hash: Hash32::default(),
            family: 0b101,
            units: 3,
            unsafe_units: 1,
            size_bound: 99,
        },
        term: t.clone(),
    };
    let b = artefact::Artefact::decode(&a.encode()).unwrap();
    assert_eq!(b.term, t);
    assert_eq!(b.header.units, 3);
}

#[test]
fn holes_only_inside_contexts() {
    // a hole at top level is not a deployable term
    let bad = Term::atom(Atom::m(CName::hole(0), CName::nil()));
    assert!(!bad.is_closed());
    assert_eq!(Term::decode(bad.encode()).unwrap_err(), DecodeError::HoleOutsideContext);
    // bound by a context it is fine
    let good = Term::atom(Atom::inst(nm(0), nm(1), bad.clone()));
    assert!(good.is_closed());
    // an inner context's hole(1) refers to the outer context
    let inner = Term::atom(Atom::inst(nm(0), nm(1), Term::atom(Atom::fw(CName::hole(0), CName::hole(1)))));
    assert_eq!(inner.free_holes(), 1);
}

#[test]
fn filling_respects_nesting() {
    let v = nm(5);
    let w = nm(6);
    let inner_ctx = Term::atom(Atom::fw(CName::hole(0), CName::hole(1)));
    let outer = Term::from_atoms(vec![Atom::m(hole_leaf(0, &[true]), CName::nil()), Atom::inst(nm(0), nm(1), inner_ctx)]);
    let f = fill(&outer, &v);
    assert!(f.is_closed());
    let inst = f.atoms().iter().find(|a| a.shape() == Shape::Inst).unwrap();
    // the inner context keeps its own hole, and the outer hole is now v
    assert_eq!(inst.context().unwrap(), &Term::atom(Atom::fw(CName::hole(0), v.clone())));
    assert_eq!(fill(inst.context().unwrap(), &w), Term::atom(Atom::fw(w, v.clone())));
    assert!(f.atoms().iter().any(|a| a.shape() == Shape::M && a.names()[0] == leaf(&v, &[true])));
}

#[test]
fn leaves_are_injective_and_disjoint() {
    let mut a = Allocator::new();
    let bound = nm(7);
    let s = a.root(&bound);
    let t = a.root(&bound);
    assert!(Allocator::check(&s, &bound, &[t.clone()]).is_ok());
    assert_eq!(Allocator::check(&leaf(&s, &[true]), &bound, &[s.clone()]), Err(AddrError::Overlap));
    let paths: Vec<Vec<bool>> = (1..40).map(gamma).collect();
    let mut seen = std::collections::HashSet::new();
    for p in &paths {
        assert!(seen.insert(leaf(&s, p)));
        assert!(leaf(&s, p).size() > bound.size());
        assert!(is_address(&leaf(&s, p)));
        assert_eq!(spine(&leaf(&s, p)), (s.clone(), p.clone()));
    }
    for (i, a) in paths.iter().enumerate() {
        for (j, b) in paths.iter().enumerate() {
            if i != j {
                assert!(!b.starts_with(a), "gamma is prefix-free");
            }
        }
    }
    assert!(!is_address(&r_star()) && !is_address(&CName::nil()));
}

#[test]
fn rule_table_is_generated() {
    let t = rules();
    assert_eq!(t.len(), RULE_COUNT);
    // seven class-(a) routing rules, quote, eval (c), make_| + family + 2 erecting (b)
    assert_eq!(t.iter().filter(|r| r.class == RuleClass::A).count(), 7);
    assert_eq!(t.iter().filter(|r| r.class == RuleClass::C).count(), 1);
    assert_eq!(t.iter().filter(|r| r.class == RuleClass::B).count(), 1 + BASE + 2);
    // every family member has one premise per built argument
    for r in t {
        if let Some(b) = r.consumer.builds() {
            assert_eq!(r.premises.len(), b.arity());
        }
    }
    let w = weights();
    assert_eq!(w[Shape::M.ix()], 0);
    assert_eq!(w[Shape::Bl.ix()], 2);
    assert_eq!(w[Shape::Inst.ix()], 1);
}

#[test]
fn conflation_is_unrepresentable() {
    // Q | m(@Q, @P) has no rule: e is an atom, not the drop (Obl. 11.8).
    let q = Term::atom(Atom::fw(nm(1), nm(2)));
    let p = Term::atom(Atom::m(nm(3), CName::nil()));
    let t = Term::par(vec![q.clone(), Term::atom(Atom::m(CName::quote(q), CName::quote(p)))]);
    // no e atom anywhere, so no eval redex can exist
    assert!(t.atoms().iter().all(|a| a.shape() != Shape::E));
}
