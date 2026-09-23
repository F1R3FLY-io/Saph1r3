use super::*;
use f1r3comb_term::addr::{f_star, r_star};
use f1r3comb_term::artefact::Scheme;

fn ok(src: &str) -> Compiled {
    match compile_source(src) {
        Ok(c) => c,
        Err(ds) => panic!("{src}: {}", ds.iter().map(|d| d.to_string()).collect::<Vec<_>>().join("; ")),
    }
}

fn codes(src: &str) -> Vec<&'static str> {
    match compile_source(src) {
        Ok(c) => c.warnings.iter().map(|d| d.code).collect(),
        Err(ds) => ds.iter().map(|d| d.code).collect(),
    }
}

const B: &str = "@{@{Nil}!(Nil)}";
const D: &str = "@{@{@{Nil}!(Nil)}!(Nil)}";
const C: &str = "@{@{Nil}!(@{Nil}!(Nil))}";

fn w() -> String {
    format!("for(z <- {D}){{ *z | *z }} | {D}!(for(y <- {B}){{ y!(*y) }}) | {B}!(Nil) | {B}!({C}!(Nil))")
}

#[test]
fn canonical_under_par_permutation_at_both_phases() {
    let a = ok(&format!("for(y <- {B}){{ *y | y!(Nil) }} | {D}!(Nil) | for(z <- {C}){{ z!(*z) }}"));
    let b = ok(&format!("for(z <- {C}){{ z!(*z) }} | {D}!(Nil) | for(y <- {B}){{ y!(Nil) | *y }}"));
    assert!(a.ir.alpha_eq(&b.ir), "phase one");
    assert_eq!(a.term.encode(), b.term.encode(), "phase two");
    // and compiling twice gives the same bytes
    assert_eq!(ok(&w()).term.encode(), ok(&w()).term.encode());
}

#[test]
fn golden_ii_the_two_instances_of_r_are_alpha_distinct() {
    // Ex. 6.5: at phase one, the two drop sites of W instantiate the same
    // quoted code; every instance of (new c)… is α-distinct, so the stored
    // code carries a binder, not a name.
    let c = ok(&w());
    let r = ok(&format!("for(y <- {B}){{ y!(*y) }}"));
    // R's image binds its proxies: they are New variables, not names
    assert!(matches!(r.ir, f1r3comb_ir::Node::New { .. }));
    assert!(r.ir.free_vars().is_empty());
    let enc = r.ir.encode();
    assert!(enc.contains(&f1r3comb_ir::tags::NEW));
    // at phase two the artefact contains no instance name at all: R's image
    // is a unit waiting for an address
    let rt = &r.term;
    assert!(rt.atoms().iter().any(|a| a.shape() == f1r3comb_term::Shape::Inst && *a.subject() == r_star()));
    assert!(rt.atoms().iter().any(|a| a.shape() == f1r3comb_term::Shape::E && *a.subject() == f_star()));
    // and W posts one address per drop of z: two
    let text = f1r3comb_term::print::render(&c.term, &[]);
    assert_eq!(text.matches("m(r*, □").count(), 2, "{text}");
}

#[test]
fn marking_distinguishes_r_from_the_near_miss() {
    let r = ok(&format!("for(y <- {B}){{ y!(*y) }}"));
    assert_eq!((r.header.units, r.header.unsafe_units), (1, 1));
    let n = ok(&format!("for(y <- {B}){{ *y | *y }}"));
    assert_eq!((n.header.units, n.header.unsafe_units), (1, 0));
    // two occurrences inside one quotation are joined by cons_|: unsafe
    let q = ok(&format!("for(y <- {B}){{ @{{*y | *y}}!(Nil) }}"));
    assert_eq!(q.header.unsafe_units, 1);
}

#[test]
fn diagnostics() {
    assert!(codes("for(y <- x){ Nil }").contains(&"comb-open"));
    assert!(codes("new x in { x!(Nil) }").iter().any(|c| *c == "comb-level"));
    assert!(codes(&format!("for(y <- {B}){{ @{{for(z <- {C}){{ *y }}}}!(Nil) }}")).contains(&"comb-o5-unsupported"));
    assert!(codes(&format!("{B}!(Nil)")).is_empty());
    let e = compile_source_with(&format!("{B}!(Nil)"), Scheme::Curried).err().unwrap();
    assert!(e.iter().any(|d| d.code == "comb-scheme-mismatch"));
}

#[test]
fn discipline_holds_on_emitted_images_and_fails_on_a_literal_one() {
    use f1r3comb_term::addr::hole_leaf;
    use f1r3comb_term::{Atom, CName, Shape, Term};
    let mut diags = Vec::new();
    verify::check(&ok(&w()).term, &mut diags);
    assert!(diags.is_empty(), "{diags:?}");
    // the literal erection of Ex. 6.16: cons_d at static channels fed by two
    // addresses at once
    let st = |i: u32| {
        let mut c = CName::nil();
        for _ in 0..i + 3 {
            c = CName::quote(Term::atom(Atom::k(c)));
        }
        c
    };
    let lit = Term::atom(Atom::inst(
        st(0),
        st(1),
        Term::from_atoms(vec![
            Atom::m(st(2), hole_leaf(0, &[true])),
            Atom::m(st(3), hole_leaf(0, &[false])),
            Atom::cons_member(Shape::D, vec![st(9), st(2), st(3), st(4)]).unwrap(),
        ]),
    ));
    let mut diags = Vec::new();
    verify::check(&lit, &mut diags);
    assert!(diags.iter().any(|d| d.code == "comb-discipline"), "{diags:?}");
    assert!(diags.iter().any(|d| d.code == "comb-o5-address"), "{diags:?}");
}

#[test]
fn header_records_scheme_family_and_bound() {
    let c = ok(&format!("for(y <- {B}){{ @{{*y}}!(Nil) | @{{y!(Nil)}}!(Nil) }}"));
    assert_eq!(c.header.scheme, Scheme::Inst);
    assert!(c.header.family & (1 << 15) != 0 || c.header.family & 1 != 0);
    assert!(c.header.size_bound > 0);
    let p = compile_source_with(&w(), Scheme::Positional).unwrap();
    assert_eq!(p.header.scheme, Scheme::Positional);
    assert!(!p.term.atoms().iter().any(|a| a.shape() == f1r3comb_term::Shape::Inst));
}

#[test]
fn nesting_family_is_linear_at_phase_one() {
    // Req. 8.9: the nesting family of Prop. 8.2 — atoms grow linearly in d.
    let mut prev = 0u64;
    let mut deltas = Vec::new();
    for d in 1..6 {
        let mut src = String::from("Nil");
        for i in 0..d {
            src = format!("for(y{i} <- {B}){{ {src} | y{i}!(Nil) }}");
        }
        let c = ok(&src);
        let (_, her) = c.ir.counts();
        deltas.push(her.total() - prev);
        prev = her.total();
    }
    assert!(deltas.windows(2).skip(1).all(|w| w[0] == w[1]), "{deltas:?}");
}
