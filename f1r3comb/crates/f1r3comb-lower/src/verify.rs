//! Post-passes on the emitted target (F1R3Comb v0.6 Req. 5.14, 5.29, 5.31):
//! conformance to the single-address-input discipline, the (o5) address
//! check, and the static-channel check. They run in the default build and
//! fail compilation.

use crate::{Diag, Severity};
use f1r3comb_term::addr::{f_star, r_star};
use f1r3comb_term::rules::{rules_for, rules};
use f1r3comb_term::{Atom, CName, Shape, Term};
use std::collections::HashSet;

/// Every atom reachable: through stores, contexts, and quoted names.
pub fn all_atoms(t: &Term) -> Vec<Atom> {
    let mut out = Vec::new();
    let mut seen = HashSet::new();
    let mut work = vec![t.clone()];
    while let Some(t) = work.pop() {
        for a in t.atoms() {
            out.push(a.clone());
            if let Some(p) = a.store().or(a.context()) {
                work.push(p.clone());
            }
            for n in a.names() {
                if let Some(p) = n.proc_ref() {
                    if seen.insert(n.content_hash()) {
                        work.push(p.clone());
                    }
                }
            }
        }
    }
    out
}

/// A name that is (or is derived from) an instance address: a leaf beneath a
/// hole or a root.
fn address_like(n: &CName) -> bool {
    !n.is_closed() || f1r3comb_term::addr::is_address(n)
}

/// Channels statically named in the term: closed and not address-derived.
fn is_static(n: &CName) -> bool {
    n.is_closed() && !f1r3comb_term::addr::is_address(n)
}

pub fn check(t: &Term, out: &mut Vec<Diag>) {
    let atoms = all_atoms(t);
    // channels that carry addresses
    let carrying: HashSet<Vec<u8>> = atoms
        .iter()
        .filter(|a| a.shape() == Shape::M && address_like(&a.names()[1]))
        .map(|a| a.names()[0].encode().to_vec())
        .collect();
    let err = |out: &mut Vec<Diag>, code: &'static str, m: String| {
        if !out.iter().any(|d| d.code == code && d.message == m) {
            out.push(Diag { code, severity: Severity::Error, span: None, message: m });
        }
    };
    for a in &atoms {
        let s = a.shape();
        if !(s.is_constructor() || matches!(s, Shape::Inst | Shape::Cstar)) {
            continue;
        }
        let rule = rules_for(s).next().unwrap();
        let inputs: Vec<&CName> = rules()[rule].premises.iter().map(|p| &a.names()[p.subject as usize]).collect();
        // Def. 6.10: a constructor at statically named channels has at most
        // one address-carrying input.
        if inputs.iter().all(|c| is_static(c)) {
            let n = inputs.iter().filter(|c| carrying.contains(c.encode())).count();
            if n > 1 {
                err(out, "comb-discipline", format!("{} joins {n} address-carrying inputs at static channels", s.name()));
            }
        }
        // Req. 5.29: no address supplies a value to an (o5) chain
        if s.is_constructor() && inputs.iter().any(|c| carrying.contains(c.encode())) {
            err(out, "comb-o5-address", format!("an address flows into a {} of an (o5) chain", s.name()));
        }
    }
    // Req. 5.31 (as implemented): no value in the image is a static channel,
    // so no encoded context can name one.
    for a in &atoms {
        if a.shape() == Shape::M && (a.names()[1] == r_star() || a.names()[1] == f_star()) {
            err(out, "comb-static-not-fresh", "a static channel occurs as a value".into());
        }
    }
    // Req. 5.24: a context only as the third argument of inst/cstar is
    // enforced by the type; a hole outside every context is not closed.
    if !t.is_closed() {
        err(out, "comb-context-misplaced", "a hole outside every context".into());
    }
}
