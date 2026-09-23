//! `f1r3comb-check` — does the compiled program do what the source does?
//!
//! * [`reference`]: a small-step K0 reducer over K1ndl1ng normal forms
//!   (COMM: `for(y <- x){P} | x!(Q) -> P{@Q/y}`), with exhaustive
//!   exploration of every interleaving for small programs.
//! * [`observe`]: what a program shows the world — the messages it leaves on
//!   names that are encodings of source names (Obl. 11.11): allocated leaves,
//!   positional names and the static channels are excluded, since no encoded
//!   context can name them (Lem. 7.6).
//! * [`differential`]: the observation of the compiled program run on a
//!   machine must be the compiled observation of *some* final state the
//!   reference reaches.

#![forbid(unsafe_code)]

pub mod phase2exp;

use f1r3comb_term::artefact::Scheme;
use f1r3comb_term::{CName, Shape, Term};
use k1ndl1ng_norm::{Name, Node, Norm, Subst, Val};
use std::collections::{BTreeMap, HashSet};

pub mod reference {
    use super::*;

    /// All one-step successors of a closed K0 process.
    pub fn successors(p: &Norm) -> Vec<Norm> {
        let parts = p.top_level();
        let mut out = Vec::new();
        for (i, s) in parts.iter().enumerate() {
            let Node::Send { chan: sc, args, .. } = s.node() else { continue };
            for (j, r) in parts.iter().enumerate() {
                let Node::Receive { binds, body } = r.node() else { continue };
                if binds.len() != 1 || binds[0].chan != *sc {
                    continue;
                }
                let v = Val::Name(Name::Quote(args[0].clone()));
                let next = body.substitute(&Subst::from_slots(vec![v]));
                let mut rest: Vec<Norm> = parts
                    .iter()
                    .enumerate()
                    .filter(|(k, _)| *k != i && *k != j)
                    .map(|(_, x)| x.clone())
                    .collect();
                rest.push(next);
                out.push(Norm::par(rest));
            }
        }
        out.sort_by(|a, b| a.encode().cmp(b.encode()));
        out.dedup_by(|a, b| a.encode() == b.encode());
        out
    }

    /// Every normal form reachable from `p`, if at most `limit` states are
    /// visited; `None` if the exploration is cut off.
    pub fn finals(p: &Norm, limit: usize) -> Option<Vec<Norm>> {
        let mut seen: HashSet<Vec<u8>> = HashSet::new();
        let mut work = vec![p.clone()];
        let mut out: BTreeMap<Vec<u8>, Norm> = BTreeMap::new();
        while let Some(q) = work.pop() {
            if !seen.insert(q.encode().to_vec()) {
                continue;
            }
            if seen.len() > limit {
                return None;
            }
            let next = successors(&q);
            if next.is_empty() {
                out.insert(q.encode().to_vec(), q);
            }
            work.extend(next);
        }
        Some(out.into_values().collect())
    }
}

/// An observation: messages on source names, as (channel, payload) content
/// hashes with multiplicity.
pub type Obs = BTreeMap<(Vec<u8>, Vec<u8>), u32>;

fn is_apparatus(n: &CName) -> bool {
    f1r3comb_lower::is_apparatus(n)
}

/// The observation of a target term.
pub fn observe(t: &Term) -> Obs {
    let mut o = Obs::new();
    for a in t.atoms() {
        if a.shape() == Shape::M && !is_apparatus(&a.names()[0]) {
            let k = (a.names()[0].encode().to_vec(), a.names()[1].encode().to_vec());
            *o.entry(k).or_default() += 1;
        }
    }
    o
}

#[derive(Clone, Debug)]
pub enum Verdict {
    /// The machine's observation matches a reference final state.
    Agree { finals: usize },
    /// It matches none.
    Disagree { finals: usize, got: Obs },
    /// The reference exploration was cut off.
    Unexplored,
    /// The machine did not reach quiescence.
    NoQuiescence(String),
}

impl Verdict {
    pub fn ok(&self) -> bool {
        matches!(self, Verdict::Agree { .. })
    }
}

/// The compiled observations of the reference's final states.
pub fn expected(src: &Norm, limit: usize) -> Option<Vec<Obs>> {
    let fs = reference::finals(src, limit)?;
    Some(
        fs.iter()
            .map(|f| observe(&f1r3comb_lower::compile_norm(f).expect("finals of a K0 program compile").term))
            .collect(),
    )
}

/// Compare a machine run's final term with the reference.
pub fn judge(src: &Norm, run: &f1r3comb_par::RunReport, limit: usize) -> Verdict {
    if run.stop != f1r3comb_par::Stop::Quiescent {
        return Verdict::NoQuiescence(format!("{:?}", run.stop));
    }
    let Some(exp) = expected(src, limit) else { return Verdict::Unexplored };
    let got = observe(&run.final_term);
    if exp.contains(&got) {
        Verdict::Agree { finals: exp.len() }
    } else {
        Verdict::Disagree { finals: exp.len(), got }
    }
}

pub fn compile(src: &str, scheme: Scheme) -> f1r3comb_lower::Compiled {
    f1r3comb_lower::compile_source_with(src, scheme).unwrap_or_else(|ds| {
        panic!("{src}: {}", ds.iter().map(|d| d.to_string()).collect::<Vec<_>>().join("; "))
    })
}

/// Deploy at the allocator's next root and run on the host.
pub fn run_host(term: &Term, seed: u64, alloc: &mut f1r3comb_term::addr::Allocator) -> f1r3comb_par::RunReport {
    let (t, _) = f1r3comb_par::deploy(term, alloc);
    let cfg = f1r3comb_par::Config { seed, ..Default::default() };
    f1r3comb_par::run(&t, &cfg, &mut f1r3comb_par::Host)
}

/// Compile `src` under `scheme`, run it on the host for each seed, and judge.
pub fn differential_with(src: &str, scheme: Scheme, seeds: &[u64]) -> Vec<Verdict> {
    let c = compile(src, scheme);
    let mut alloc = f1r3comb_term::addr::Allocator::new();
    seeds.iter().map(|s| judge(&c.source, &run_host(&c.term, *s, &mut alloc), 20_000)).collect()
}

pub fn differential_host(src: &str, seeds: &[u64]) -> Vec<Verdict> {
    differential_with(src, Scheme::Inst, seeds)
}

/// Four distinct channels. (`@{Nil|Nil}` is `@{Nil}`: the monoid laws.)
pub const A: &str = "@{Nil}";
pub const B: &str = "@{@{Nil}!(Nil)}";
pub const C: &str = "@{@{Nil}!(@{Nil}!(Nil))}";
pub const D: &str = "@{@{@{Nil}!(Nil)}!(Nil)}";

/// Expand `$A`..`$D` to the four channels.
pub fn chans(s: &str) -> String {
    s.replace("$A", A).replace("$B", B).replace("$C", C).replace("$D", D)
}

/// The differential corpus, before [`chans`]: closed K0 programs exercising
/// every occurrence kind, nesting, dynamic names, received code and races.
pub const CORPUS_SRC: &[&str] = &[
    "$A!(Nil)",
    "for(y <- $A){ Nil } | $A!(Nil)",
    // (o4)
    "for(y <- $A){ *y } | $A!($B!(Nil))",
    // (o2)
    "for(y <- $A){ y!(Nil) } | $A!($B!(Nil))",
    // (o3), static channel
    "for(y <- $A){ $C!(*y) } | $A!($B!(Nil))",
    // (o1), then a message on the received name
    "for(y <- $A){ for(z <- y){ *z } } | $A!(Nil) | @{Nil}!($C!(Nil))",
    // several occurrences of one binder
    "for(y <- $A){ y!(Nil) | *y | $C!(*y) } | $A!($B!(Nil))",
    // nested inputs: the inner link waits for the inner input
    "for(y <- $A){ for(z <- $B){ y!(*z) } } | $A!($C!(Nil)) | $B!(Nil)",
    "for(y <- $A){ for(z <- $B){ y!(*z) } } | $A!($C!(Nil))",
    // (o3) against a dynamic channel (relay)
    "for(y <- $A){ for(z <- $B){ z!(*y) } } | $A!($C!(Nil)) | $B!($D!(Nil))",
    // dynamic names (o5), a chain of cons_| and cons_m
    "for(y <- $A){ @{*y}!(Nil) } | $A!($B!(Nil))",
    "for(y <- $A){ @{y!(Nil)}!(Nil) } | $A!($B!(Nil))",
    "for(y <- $A){ $C!(y!(Nil) | $D!(Nil)) } | $A!(Nil)",
    "for(y <- $A){ for(z <- @{*y | *y}){ *z } } | $A!(Nil) | @{Nil | Nil}!($D!(Nil))",
    // races
    "for(y <- $A){ $C!(*y) } | for(y <- $A){ $D!(*y) } | $A!(Nil)",
    "for(y <- $A){ $C!(*y) } | $A!(Nil) | $A!($B!(Nil))",
    // received code with an input, run once and run twice
    "for(y <- $A){ *y } | $A!(for(z <- $B){ $C!(*z) }) | $B!(Nil)",
    "for(y <- $A){ *y | *y } | $A!(for(z <- $B){ $C!(*z) }) | $B!(Nil) | $B!($D!(Nil))",
    // a received process that itself receives code and runs it
    "for(y <- $A){ *y } | $A!(for(z <- $B){ *z }) | $B!(for(u <- $C){ $D!(*u) }) | $C!(Nil)",
];

pub fn corpus() -> Vec<String> {
    CORPUS_SRC.iter().map(|s| chans(s)).collect()
}


#[cfg(test)]
mod tests;

/// Compile `src`, run it on the GPU for each seed, check the device run's
/// trace equals the host's, and judge against the reference.
#[cfg(feature = "gpu")]
pub fn differential_gpu(g: &mut f1r3comb_gpu::device::Gpu, src: &str, seeds: &[u64]) -> Vec<Verdict> {
    let c = f1r3comb_lower::compile_source(src).expect("compiles");
    let mut alloc = f1r3comb_term::addr::Allocator::new();
    let (t, _) = f1r3comb_par::deploy(&c.term, &mut alloc);
    seeds
        .iter()
        .map(|s| {
            let cfg = f1r3comb_par::Config { seed: *s, hash_states: true, ..Default::default() };
            let host = f1r3comb_par::run(&t, &cfg, &mut f1r3comb_par::Host);
            let dev = f1r3comb_par::run(&t, &cfg, g);
            assert_eq!(host.steps, dev.steps, "{src}: device trace differs from host (seed {s})");
            judge(&c.source, &dev, 20_000)
        })
        .collect()
}

/// The golden text for a program (F1R3Comb v0.6 Req. 8.8): source, phase-one
/// IR with its hash, target with its hash and encoding, the deployed root,
/// and the reduction sequence under maximal progress, seed 0, with per-step
/// rule counts, names built and state hashes.
pub fn golden(src: &str, max_steps: u64) -> String {
    use std::fmt::Write;
    let c = compile(src, Scheme::Inst);
    let mut s = String::new();
    let _ = writeln!(s, "# source\n{src}\n");
    let _ = writeln!(s, "# phase one (RC^nu), hash {}\n{}", c.ir.hash().hex(), c.ir.render());
    let _ = writeln!(s, "# phase two\n{}", f1r3comb_term::print::render(&c.term, &[]));
    let _ = writeln!(s, "# encoding ({} bytes)", c.term.encode().len());
    for chunk in c.term.encode().chunks(48) {
        let _ = writeln!(s, "{}", chunk.iter().map(|b| format!("{b:02x}")).collect::<String>());
    }
    let (t, root) = f1r3comb_par::deploy(&c.term, &mut f1r3comb_term::addr::Allocator::new());
    let _ = writeln!(s, "\n# deploy: root {}", root.map(|r| r.content_hash().hex()).unwrap_or_default());
    let cfg = f1r3comb_par::Config { seed: 0, max_steps, hash_states: true, ..Default::default() };
    let r = f1r3comb_par::run(&t, &cfg, &mut f1r3comb_par::Host);
    let _ = writeln!(s, "# reduction (maximal progress, seed 0)");
    for st in &r.steps {
        let rules: Vec<String> = st
            .per_rule
            .iter()
            .enumerate()
            .filter(|(_, n)| **n > 0)
            .map(|(i, n)| format!("{}:{n}", f1r3comb_term::rules::rules()[i].name))
            .collect();
        let _ = writeln!(s, "{:4} fired {:3} names {:3}  {:40} {}", st.step, st.fired, st.names_built, rules.join(" "),
            st.state_hash.map(|h| h.hex()[..16].to_string()).unwrap_or_default());
    }
    let _ = writeln!(s, "# stop {:?}, {} steps, {} firings, final {}", r.stop, r.steps.len(), r.fired, r.final_hash().hex());
    let lbl = |e: &[u8]| format!("#{}", &f1r3comb_term::blake2b(e).hex()[..8]);
    let obs: Vec<String> = observe(&r.final_term).iter().map(|((a, b), n)| format!("{}!{} x{n}", lbl(a), lbl(b))).collect();
    let _ = writeln!(s, "# observation: {}", obs.join(", "));
    s
}
