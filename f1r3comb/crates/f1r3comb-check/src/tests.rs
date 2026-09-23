use super::*;

#[test]
fn reference_reduces() {
    let n = k1ndl1ng_norm_of("for(y <- @{Nil}){ *y } | @{Nil}!(@{Nil}!(Nil))");
    let fs = reference::finals(&n, 100).unwrap();
    assert_eq!(fs.len(), 1);
    assert_eq!(fs[0].encode(), k1ndl1ng_norm_of("@{Nil}!(Nil)").encode());
}

pub(crate) fn k1ndl1ng_norm_of(src: &str) -> Norm {
    f1r3comb_lower::compile_source(src).unwrap().source
}

#[test]
fn corpus_on_host() {
    let mut bad = Vec::new();
    for src in corpus() {
        for v in differential_host(&src, &[0, 1, 2, 3]) {
            if !v.ok() {
                bad.push(format!("{src}: {v:?}"));
            }
        }
    }
    assert!(bad.is_empty(), "{}", bad.join("\n"));
}


#[test]
fn judge_detects_a_wrong_program() {
    let c = compile("for(y <- @{Nil}){ y!(Nil) } | @{Nil}!(@{Nil}!(Nil))", Scheme::Inst);
    let r = run_host(&c.term, 0, &mut f1r3comb_term::addr::Allocator::new());
    assert!(judge(&c.source, &r, 1000).ok());
    let other = k1ndl1ng_norm_of("for(y <- @{Nil}){ *y } | @{Nil}!(@{Nil}!(Nil))");
    assert!(matches!(judge(&other, &r, 1000), Verdict::Disagree { .. }));
    assert!(!observe(&r.final_term).is_empty());
}

/// Draft 3 §6.1's W.
/// a = $B, b = $D, X = Nil, Y = $C!(Nil): @X, @Y, a, b pairwise distinct.
pub const W: &str = "for(z <- $D){ *z | *z } | $D!(for(y <- $B){ y!(*y) }) | $B!(Nil) | $B!($C!(Nil))";

/// The three-instance variant (Obl. 11.1).
pub const W3: &str = "for(z <- $D){ *z | *z | *z } | $D!(for(y <- $B){ y!(*y) }) | $B!(Nil) | $B!($C!(Nil)) | $B!($A!($A!(Nil)))";

/// The near miss of Ex. 6.21: non-linear, but not deciding.
pub const NEAR_MISS: &str = "for(z <- $D){ *z | *z } | $D!(for(y <- $B){ *y | *y }) | $B!($C!(Nil)) | $B!($A!($A!(Nil)))";

fn tally(v: &[Verdict]) -> (usize, usize) {
    let agree = v.iter().filter(|x| x.ok()).count();
    let disagree = v.iter().filter(|x| matches!(x, Verdict::Disagree { .. })).count();
    assert_eq!(agree + disagree, v.len(), "{v:?}");
    (agree, disagree)
}

#[test]
fn w_regression_under_inst() {
    let seeds: Vec<u64> = (0..100).collect();
    for src in [W, W3] {
        let (_, bad) = tally(&differential_with(&chans(src), Scheme::Inst, &seeds));
        assert_eq!(bad, 0, "{src}");
    }
    // and explicitly: the crossed residue m(@X, Y) never appears
    let c = compile(&chans(W), Scheme::Inst);
    let x = f1r3comb_lower::compile_name(&k1ndl1ng_norm_of("Nil")).unwrap();
    let y = f1r3comb_lower::compile_name(&k1ndl1ng_norm_of(&chans("$C!(Nil)"))).unwrap();
    let mut alloc = f1r3comb_term::addr::Allocator::new();
    for s in 0..100 {
        let r = run_host(&c.term, s, &mut alloc);
        let crossed = r.final_term.atoms().iter().any(|a| a.shape() == Shape::M && a.names()[0] == x && a.names()[1] == y);
        assert!(!crossed, "seed {s}");
    }
}

#[test]
fn positional_naming_is_the_negative_control() {
    let seeds: Vec<u64> = (0..100).collect();
    let (_, bad) = tally(&differential_with(&chans(W), Scheme::Positional, &seeds));
    assert!(bad > 0, "the positional scheme should cross on W");
    let (_, bad3) = tally(&differential_with(&chans(W3), Scheme::Positional, &seeds));
    assert!(bad3 > 0);
    // the near miss crosses too, but parallel composition absorbs it
    let (_, near) = tally(&differential_with(&chans(NEAR_MISS), Scheme::Positional, &seeds));
    assert_eq!(near, 0);
    eprintln!("positional crossing on W: {bad}/100, on W3: {bad3}/100, on the near miss: {near}/100");
}

#[test]
fn marking_agrees_with_the_positional_oracle() {
    // Req. 5.17 / Obl. 11.4: R's receive is unsafe, the near miss's safe.
    let w = compile(&chans(W), Scheme::Inst);
    let n = compile(&chans(NEAR_MISS), Scheme::Inst);
    assert!(w.header.unsafe_units >= 1, "R must be unsafe");
    assert_eq!(n.header.unsafe_units, 0, "the near miss must be safe");
    // the oracle: exhaustive over schedules is replaced by 100 seeds of the
    // shared-apparatus build at two and three instances
    let seeds: Vec<u64> = (0..100).collect();
    for (src, unsafe_) in [(W, true), (W3, true), (NEAR_MISS, false)] {
        let (_, bad) = tally(&differential_with(&chans(src), Scheme::Positional, &seeds));
        assert_eq!(bad > 0, unsafe_, "{src}");
    }
}

#[test]
fn address_independence() {
    // Obl. 11.14: the corpus at two unrelated roots gives the same observations.
    for src in corpus() {
        let c = compile(&src, Scheme::Inst);
        let mut a = f1r3comb_term::addr::Allocator::new();
        let r1 = run_host(&c.term, 3, &mut a);
        let r2 = run_host(&c.term, 3, &mut a); // second root
        assert_eq!(observe(&r1.final_term), observe(&r2.final_term), "{src}");
    }
}

#[test]
fn conflation_regression_on_the_machine() {
    // Obl. 11.8: Q | m(@Q, @P) is stuck for a nontrivial Q.
    use f1r3comb_term::Atom;
    let q = Term::atom(Atom::fw(CName::nil(), f1r3comb_term::addr::k_const()));
    let p = Term::atom(Atom::m(f1r3comb_term::addr::k_const(), CName::nil()));
    let t = Term::par(vec![q.clone(), Term::atom(Atom::m(CName::quote(q), CName::quote(p)))]);
    let r = f1r3comb_par::run(&t, &Default::default(), &mut f1r3comb_par::Host);
    assert_eq!(r.fired, 0);
    assert_eq!(r.final_term, t);
}

#[test]
fn freshness_adversarial_source_name() {
    // Obl. 11.9: @(x!(Nil)) encodes as L[[x]], a leaf of [[x]] read as an
    // address; roots are larger than every name of the image, so no leaf of
    // a deploy is ever such a name, and it is not apparatus.
    let c = compile("for(y <- @{@{Nil}!(Nil)}){ *y } | @{@{Nil}!(Nil)}!(@{Nil}!(Nil))", Scheme::Inst);
    let adversarial = f1r3comb_lower::compile_name(&k1ndl1ng_norm_of("@{Nil}!(Nil)")).unwrap();
    assert!(!is_apparatus(&adversarial));
    let mut alloc = f1r3comb_term::addr::Allocator::new();
    let (_, root) = f1r3comb_par::deploy(&c.term, &mut alloc);
    let root = root.unwrap();
    assert!(root.size() as u64 > c.header.size_bound);
    assert!(f1r3comb_term::addr::Allocator::check(&root, &f1r3comb_term::addr::largest_name(&c.term), &[]).is_ok());
}

mod phase2 {
    use crate::phase2exp::*;

    #[test]
    fn e5_three_arms() {
        let mut report = String::new();
        for arm in [Arm::Literal, Arm::Curried, Arm::Inst] {
            for n in [2usize, 4, 8, 16] {
                for par in [true, false] {
                    let runs: Vec<(u64, usize, usize)> = (0..50).map(|s| e5(arm, n, s, par)).collect();
                    let bad = runs.iter().filter(|r| r.2 > 0).count();
                    let steps = runs.iter().map(|r| r.0).sum::<u64>() as f64 / 50.0;
                    report += &format!("E5 {arm:?} n={n} {} steps={steps:.2} runs-with-mixing={bad}/50\n",
                        if par { "parallel" } else { "sequential" });
                    match arm {
                        Arm::Literal => {
                            if n >= 4 {
                                assert!(bad > 0, "literal must mix at n={n}");
                            }
                        }
                        _ => assert_eq!(bad, 0, "{arm:?} n={n} must not mix"),
                    }
                    if par {
                        let expect = match arm {
                            Arm::Literal => 5.0,
                            Arm::Curried => 3.0,
                            Arm::Inst => 2.0,
                        };
                        assert_eq!(steps, expect, "{arm:?} parallel steps");
                    }
                }
            }
        }
        eprint!("{report}");
    }

    #[test]
    fn e6_dx_laws() {
        let (ds, dn) = e6(Arm::Inst, 200, 0, true);
        assert_eq!(ds.len(), 199);
        let steps: std::collections::BTreeSet<u64> = ds.iter().copied().collect();
        let names: std::collections::BTreeSet<u64> = dn.iter().copied().collect();
        let (ss, _) = e6(Arm::Inst, 200, 0, false);
        let mean = ss.iter().sum::<u64>() as f64 / ss.len() as f64;
        let (cds, cdn) = e6(Arm::Curried, 200, 0, true);
        let csteps: std::collections::BTreeSet<u64> = cds.iter().copied().collect();
        let cnames: std::collections::BTreeSet<u64> = cdn.iter().copied().collect();
        let (css, _) = e6(Arm::Curried, 200, 0, false);
        let cmean = css.iter().sum::<u64>() as f64 / css.len() as f64;
        eprintln!("E6 inst: parallel steps/unfolding {steps:?}, sequential mean {mean:.2}, names/unfolding {names:?}");
        eprintln!("E6 curried: parallel steps/unfolding {csteps:?}, sequential mean {cmean:.2}, names/unfolding {cnames:?}");
        // phase2-results.txt: inst 7 steps, 16 names, sequential 10; curried 10, 22, 32
        assert_eq!(steps.into_iter().collect::<Vec<_>>(), vec![7]);
        assert_eq!(csteps.into_iter().collect::<Vec<_>>(), vec![10]);
        // constant per unfolding over 200: prefix sharing (Req. 9.8)
        assert_eq!(names.into_iter().collect::<Vec<_>>(), vec![16]);
        assert_eq!(cnames.into_iter().collect::<Vec<_>>(), vec![22]);
        assert!((mean - 10.0).abs() < 1.0 && (cmean - 32.0).abs() < 3.0);
    }

    #[test]
    fn e7_curried_stores_mix_inst_stores_do_not() {
        let inst2: usize = (0..50).map(|s| e7(Arm::Inst, 2, s)).sum();
        let inst8: usize = (0..50).map(|s| e7(Arm::Inst, 8, s)).sum();
        let cur2: usize = (0..50).map(|s| e7(Arm::Curried, 2, s)).sum();
        let cur8: usize = (0..50).map(|s| e7(Arm::Curried, 8, s)).sum();
        eprintln!("E7 mixed stores: inst {inst2}/100, {inst8}/400; curried {cur2}/100, {cur8}/400");
        assert_eq!(inst2 + inst8, 0);
        assert!(cur2 > 0 && cur8 > 0);
    }
}

#[test]
fn golden_iii_dx_compiled() {
    use crate::phase2exp::compiled_dx;
    let (ds, dn) = compiled_dx(200, 0, true);
    let (ss, _) = compiled_dx(200, 0, false);
    let steps: std::collections::BTreeSet<u64> = ds.iter().copied().collect();
    let names: std::collections::BTreeSet<u64> = dn.iter().copied().collect();
    let mean = ss.iter().sum::<u64>() as f64 / ss.len() as f64;
    eprintln!("compiled D_x: parallel steps/unfolding {steps:?}, sequential mean {mean:.2}, names/unfolding {names:?}");
    assert_eq!(ds.len(), 199);
    assert_eq!(steps.len(), 1, "constant steps per unfolding, independent of address depth");
    assert_eq!(names.len(), 1, "constant names per unfolding");
}

#[cfg(feature = "gpu")]
#[test]
fn corpus_on_gpu() {
    let mut g = match f1r3comb_gpu::device::Gpu::new() {
        Ok(g) => g,
        Err(e) => {
            eprintln!("skipping: {e}");
            return;
        }
    };
    for src in corpus() {
        for v in differential_gpu(&mut g, &src, &[0, 1]) {
            assert!(v.ok(), "{src}: {v:?}");
        }
    }
    // W on the device, and the step traces equal the host's
    for v in differential_gpu(&mut g, &chans(W), &(0..16).collect::<Vec<_>>()) {
        assert!(v.ok(), "W: {v:?}");
    }
}
