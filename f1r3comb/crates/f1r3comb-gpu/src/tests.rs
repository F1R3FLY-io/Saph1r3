use super::*;
use f1r3comb_term::addr::hole_leaf;
use f1r3comb_term::{Atom, CName};

fn seed_one(c: &CName) -> CName {
    CName::quote(Term::atom(Atom::k(c.clone())))
}

#[test]
fn kernels_parse_as_wgsl_and_bundle_round_trips() {
    let k = wgsl::kernels();
    let module = naga::front::wgsl::parse_str(&k).expect("kernels parse");
    naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::empty())
        .validate(&module)
        .expect("kernels validate");
    for e in wgsl::ENTRY_POINTS {
        assert!(k.contains(&format!("fn {e}(")), "{e}");
    }
    let a = seed_one(&CName::nil());
    let b = seed_one(&a);
    // a unit waiting at b for an address, and the address
    let ctx = Term::from_atoms(vec![Atom::fw(hole_leaf(0, &[true]), a.clone()), Atom::m(hole_leaf(0, &[true]), CName::nil())]);
    let t = Term::from_atoms(vec![
        Atom::fw(CName::nil(), a.clone()),
        Atom::m(CName::nil(), a.clone()),
        Atom::inst(b.clone(), seed_one(&b), ctx),
        Atom::e(seed_one(&b)),
        Atom::m(b.clone(), seed_one(&seed_one(&b))),
    ]);
    let dir = std::env::temp_dir().join(format!("f1r3comb-bundle-{}", std::process::id()));
    let r = write_bundle(&dir, &t, None, &Config::default()).unwrap();
    assert_eq!(r.fired, 4);
    let (back, same) = read_bundle(&dir).unwrap();
    assert_eq!(back, t);
    assert!(same);
    let m = std::fs::read_to_string(dir.join("manifest.json")).unwrap();
    assert!(m.contains("\"make_s\""));
    assert!(m.contains("\"inst\""));
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn a_device_rejects_curried_artefacts() {
    let a = seed_one(&CName::nil());
    let t = Term::atom(Atom::cstar(a.clone(), seed_one(&a), Term::atom(Atom::k(hole_leaf(0, &[false])))));
    assert!(device_accepts(&t).unwrap_err().contains("comb-mat-scheme"));
    let dir = std::env::temp_dir().join(format!("f1r3comb-bundle-c-{}", std::process::id()));
    assert!(write_bundle(&dir, &t, None, &Config::default()).is_err());
}
