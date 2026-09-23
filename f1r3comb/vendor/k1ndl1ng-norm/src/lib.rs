//! `k1ndl1ng-norm` — binder resolution, de Bruijn assignment, ordering of
//! parallel components, the normal form, its encoding, content hashing,
//! substitution and shifting (K1ndl1ng spec §7).
//!
//! Name equality costs a 32-byte comparison, which is the point of the
//! exercise. The crate does not depend on the parser (Req. 3.5): any
//! well-formed tree will do, however it was built.

#![forbid(unsafe_code)]

pub mod build;
pub mod hash;
pub mod json;
pub mod rebuild;
pub mod show;
pub mod term;

pub use build::{normalise, Diag, FreeName, Normalised, Options, Sort};
pub use hash::{Blake2b256, Digest32, Hash32};
pub use json::{tree_to_string, write_tree, Budget, Buf};
pub use rebuild::{free_key, receive_permuting, Subst, Val};
pub use show::{show, show_pretty, unnormalise};
pub use term::{BindKind, DecodeError, NBind, Name, Node, Norm};

/// Minting policy for the unforgeable names a `new` introduces.
/// `Sequential` is a pure function of the binder's position, so a
/// K1ndl1ng-only pipeline is deterministic and reproducible. It is *not*
/// `f1r3node`'s deploy-seeded derivation and must not be presented as such.
pub trait NameMinter {
    fn mint(&mut self, path: &BinderPath) -> [u8; 32];
}

/// Where a `new` binder sits: the sequence of choices taken from the root, plus
/// the slot within the binder.
#[derive(Clone, Debug, Default)]
pub struct BinderPath {
    pub steps: Vec<u32>,
    pub slot: u32,
}

#[derive(Default)]
pub struct Sequential {
    counter: u64,
}

impl NameMinter for Sequential {
    fn mint(&mut self, path: &BinderPath) -> [u8; 32] {
        let mut pre = Vec::with_capacity(16 + 4 * path.steps.len());
        pre.extend_from_slice(b"k1ndl1ng/seq/");
        pre.extend_from_slice(&self.counter.to_le_bytes());
        for s in &path.steps {
            pre.extend_from_slice(&s.to_le_bytes());
        }
        pre.extend_from_slice(&path.slot.to_le_bytes());
        self.counter += 1;
        hash::blake2b_256(&pre).0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use k1ndl1ng_parse::{parse, Options as POpts};

    fn norm(src: &str) -> Norm {
        let p = parse(src, &POpts::default());
        assert!(p.ok(), "parse failed for {src}: {:?}", p.diags);
        normalise(&p.tree, &Options::default())
            .unwrap_or_else(|d| panic!("normalise failed for {src}: {d:?}"))
            .term
    }

    fn enc(src: &str) -> Vec<u8> {
        norm(src).encode().to_vec()
    }

    #[test]
    fn par_is_commutative_and_associative() {
        assert_eq!(enc("x!(Nil) | y!(Nil)"), enc("y!(Nil) | x!(Nil)"));
        assert_eq!(enc("(a!(Nil) | b!(Nil)) | c!(Nil)"), enc("a!(Nil) | (b!(Nil) | c!(Nil))"));
    }

    #[test]
    fn nil_is_the_unit() {
        assert_eq!(enc("x!(Nil) | Nil"), enc("x!(Nil)"));
        assert_eq!(enc("Nil | Nil"), enc("Nil"));
    }

    #[test]
    fn alpha_equivalence_is_absorbed() {
        assert_eq!(enc("for(y <- x){ *y }"), enc("for(z <- x){ *z }"));
        assert_eq!(enc("new a in { a!(Nil) }"), enc("new b in { b!(Nil) }"));
    }

    #[test]
    fn quote_unquote_rewrites() {
        assert_eq!(enc("*@{x!(Nil)}"), enc("x!(Nil)"));
        assert_eq!(enc("@{*x}!(Nil)"), enc("x!(Nil)"));
    }

    #[test]
    fn different_terms_differ() {
        assert_ne!(enc("x!(Nil)"), enc("x!!(Nil)"));
        assert_ne!(enc("for(y <- x){Nil}"), enc("for(y <= x){Nil}"));
        assert_ne!(enc("x!(Nil)"), enc("x!(x!(Nil))"));
    }

    #[test]
    fn free_names_are_levels_not_identifiers() {
        // Deliberate, and what the specification says: the encoding carries
        // levels, never identifiers, so two terms that differ only by renaming
        // their free names encode identically. Their free lists differ, and an
        // executive grounds free names before running a term.
        assert_eq!(enc("x!(Nil)"), enc("y!(Nil)"));
        let a = norm("x!(Nil)").ground_free_by(&["x".to_string()]);
        let b = norm("y!(Nil)").ground_free_by(&["y".to_string()]);
        assert_ne!(a.encode(), b.encode());
    }

    #[test]
    fn joins_are_ordered_but_slots_follow() {
        // The two receipts differ only in the order the binds are written, so
        // sorting the binds must carry the body's binder slots with it.
        assert_eq!(
            enc("new a, b in { for(y <- a & z <- b){ y!(*z) } }"),
            enc("new a, b in { for(z <- b & y <- a){ y!(*z) } }")
        );
        assert_ne!(
            enc("new a, b in { for(y <- a & z <- b){ y!(*z) } }"),
            enc("new a, b in { for(y <- a & z <- b){ z!(*y) } }")
        );
    }

    #[test]
    fn free_names_are_reported_in_order() {
        let p = parse("q!(Nil) | p!(Nil)", &POpts::default());
        let n = normalise(&p.tree, &Options::default()).unwrap();
        let idents: Vec<String> = n.free.iter().map(|f| f.ident.clone()).collect();
        // Levels are assigned by first occurrence in the *ordered* term, and
        // the two components here are equal up to their free levels, so the
        // source order survives as the level order.
        assert_eq!(n.free.len(), 2);
        assert_eq!(idents, vec!["q".to_string(), "p".to_string()]);
        let p2 = parse("p!(Nil) | q!(Nil)", &POpts::default());
        let n2 = normalise(&p2.tree, &Options::default()).unwrap();
        assert_eq!(n.term.encode(), n2.term.encode());
        assert_eq!(
            n2.free.iter().map(|f| f.ident.clone()).collect::<Vec<_>>(),
            vec!["p".to_string(), "q".to_string()]
        );
    }

    #[test]
    fn encode_decode_round_trip() {
        for src in [
            "Nil",
            "x!(Nil)",
            "for(y <- x){ *y }",
            "new a, b in { a!(*b) | for(@P <= a){ b!!(P) } }",
            "for(y <- a & @Q <<- b){ y!(Q) } | a!(*@{Nil})",
        ] {
            let t = norm(src);
            let back = Norm::decode(t.encode()).expect("decode");
            assert_eq!(t.encode(), back.encode(), "round trip failed for {src}");
            assert_eq!(t.hash(), back.hash());
        }
    }

    #[test]
    fn decode_is_total_on_garbage() {
        for bytes in [
            vec![],
            vec![0xff],
            vec![0x01, 0x05],
            vec![0x02, 0x10, 0x00],
            vec![0x03, 0x01, 0x20, 0x00],
            vec![0x13, 0x00],
        ] {
            let _ = Norm::decode(&bytes);
        }
    }

    #[test]
    fn substitution_replaces_the_innermost_group() {
        // for(y <- x){ y!(Nil) }, substituting the name @{z!(Nil)} for y.
        let t = norm("for(y <- x){ y!(Nil) }");
        let body = match t.node() {
            Node::Receive { body, .. } => body.clone(),
            _ => panic!("expected a receive"),
        };
        let val = Name::Quote(norm("z!(Nil)"));
        let out = body.substitute(&Subst::from_slots(vec![Val::Name(val)]));
        assert_eq!(out.encode(), norm("@{z!(Nil)}!(Nil)").encode());
    }

    #[test]
    fn substitution_renormalises() {
        // Substituting into a par re-sorts it: the result is in normal form.
        let t = norm("for(y <- x){ y!(Nil) | @{Nil}!(*@{Nil}) }");
        let body = match t.node() {
            Node::Receive { body, .. } => body.clone(),
            _ => panic!(),
        };
        let out = body.substitute(&Subst::from_slots(vec![Val::Name(Name::Quote(Norm::nil()))]));
        let direct = norm("@{Nil}!(Nil) | @{Nil}!(Nil)");
        assert_eq!(out.encode(), direct.encode());
    }

    #[test]
    fn static_checks_fire() {
        let cases = [
            ("for(@{x | x} <- c){ Nil }", "dup-pattern-var"),
            ("*_", "eval-wild"),
            ("new a in { for(a <- c){ Nil } }", "new-as-pattern"),
        ];
        for (src, code) in cases {
            let p = parse(src, &POpts::default());
            let r = normalise(&p.tree, &Options::default());
            match r {
                Err(ds) => assert!(
                    ds.iter().any(|d| d.code == code),
                    "{src}: expected {code}, got {ds:?}"
                ),
                Ok(_) => panic!("{src} should not normalise"),
            }
        }
    }

    #[test]
    fn closed_terms_can_be_required() {
        let p = parse("x!(Nil)", &POpts::default());
        let opts = Options {
            require_closed: true,
            ..Default::default()
        };
        assert!(normalise(&p.tree, &opts).is_err());
        let p2 = parse("new a in { a!(Nil) }", &POpts::default());
        assert!(normalise(&p2.tree, &opts).is_ok());
    }

    #[test]
    fn unnormalise_round_trips_through_the_parser() {
        for src in [
            "x!(Nil)",
            "for(y <- x){ *y }",
            "new a in { a!(Nil) | for(@P <= a){ Nil } }",
            "for(y <- a & @Q <<- b){ y!(Q) }",
        ] {
            let t = norm(src);
            let text = show(&t);
            let again = norm(&text);
            assert_eq!(
                t.encode(),
                again.encode(),
                "unnormalise/reparse changed {src} (printed as {text})"
            );
        }
    }

    #[test]
    fn idempotent_under_renormalisation() {
        let t = norm("b!(Nil) | a!(Nil) | for(y <- c){ Nil }");
        let again = norm(&show(&t));
        assert_eq!(t.encode(), again.encode());
        assert_eq!(t.hash(), again.hash());
    }

    #[test]
    fn hash_agrees_with_encoding() {
        let a = norm("a!(Nil) | b!(Nil)");
        let b = norm("b!(Nil) | a!(Nil)");
        assert_eq!(a.hash(), b.hash());
        let c = norm("a!(Nil) | b!(b!(Nil))");
        assert_ne!(a.hash(), c.hash());
    }
}
