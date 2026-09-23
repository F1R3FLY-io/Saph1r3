//! `f1r3comb-gpu` — the GPU target.
//!
//! The target is a **bundle**, a directory holding
//!
//! * `program.cmat` — the initial state in F1R3Comb-Mat's canonical matrix
//!   format (names relabelled by quotation depth, then content hash);
//! * `kernels.wgsl` — the compute kernels, generated from the rule table;
//! * `manifest.json` — format versions, the rule table, the buffer layout,
//!   the dispatch sequence of one step, and a host reference trace summary
//!   (steps, firings, final state hash under seed 0) that a device run must
//!   reproduce.
//!
//! With feature `device` (default) the crate also runs a bundle on a GPU
//! through wgpu (Vulkan, Metal or DX12). The device finds redexes in
//! canonical order, computes priorities and the maximal independent step;
//! the host commits (Mat Req. 7.1), sharing `f1r3comb-par::commit`, so the
//! device and host traces are identical by construction of the step and by
//! test.

#![forbid(unsafe_code)]

pub mod wgsl;

#[cfg(feature = "device")]
pub mod device;

use f1r3comb_mat::{cmat_decode, cmat_encode, InternTable, NameBudget, State};
use f1r3comb_par::{Config, Host, RunReport};
use f1r3comb_term::rules::{rules, Conclusion, Datum};
use f1r3comb_term::Shape;
use f1r3comb_term::{Hash32, Term};
use std::path::Path;

pub const BUNDLE_VERSION: u32 = 2;

/// Mat v0.5 Req. 10.1: a device targets w = 4 and the `inst` erection only;
/// a curried (or literal) artefact is rejected with `comb-mat-scheme`.
pub fn device_accepts(term: &Term) -> Result<(), String> {
    let mut work = vec![term.clone()];
    let mut seen = std::collections::HashSet::new();
    while let Some(t) = work.pop() {
        for a in t.atoms() {
            if a.shape() == Shape::Cstar {
                return Err("a curried (cstar) erection cannot reach a device; devices run inst only [comb-mat-scheme]".into());
            }
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
    Ok(())
}

fn hex(h: &Hash32) -> String {
    h.0.iter().map(|b| format!("{b:02x}")).collect()
}

fn esc(s: &str) -> String {
    s.replace('\\', "\\\\").replace('"', "\\\"")
}

/// The manifest, as JSON text.
pub fn manifest(term: &Term, source_hash: Option<&Hash32>, cmat: &[u8], kernels: &str, reference: &RunReport, cfg: &Config) -> String {
    let mut s = String::from("{\n");
    s += &format!("  \"format\": \"f1r3comb-gpu-bundle\",\n  \"version\": {BUNDLE_VERSION},\n");
    s += "  \"presentation\": \"A\",\n  \"shortcuts\": false,\n  \"erection\": \"inst\",\n  \"row_width\": 4,\n  \"program\": \"the deployed state: the compiled term plus its root address at r*\",\n";
    s += &format!("  \"term_hash\": \"{}\",\n", hex(&term.content_hash()));
    if let Some(h) = source_hash {
        s += &format!("  \"source_hash\": \"{}\",\n", hex(h));
    }
    s += &format!("  \"cmat_hash\": \"{}\",\n", hex(&f1r3comb_term::blake2b(cmat)));
    s += &format!("  \"kernels_hash\": \"{}\",\n", hex(&f1r3comb_term::blake2b(kernels.as_bytes())));
    s += "  \"rules\": [\n";
    for (i, r) in rules().iter().enumerate() {
        let prem: Vec<String> = r
            .premises
            .iter()
            .map(|p| format!("{{\"subject\": {}, \"datum\": \"{}\"}}", p.subject, if p.datum == Datum::Loud { "m" } else { "q" }))
            .collect();
        let class = match r.conclusion {
            Conclusion::Atoms(_) => "a",
            Conclusion::Encode { .. } => "b",
            Conclusion::Decode { .. } => "c",
        };
        s += &format!(
            "    {{\"index\": {i}, \"name\": \"{}\", \"consumer\": \"{}\", \"premises\": [{}], \"class\": \"{class}\"}}{}\n",
            r.name,
            r.consumer.name(),
            prem.join(", "),
            if i + 1 < rules().len() { "," } else { "" }
        );
    }
    s += "  ],\n";
    s += "  \"buffers\": {\n";
    s += &format!(
        "    \"meta\": {{\"binding\": 0, \"access\": \"read\", \"words\": {{\"table_len\": {}, \"column_base\": {}, \"rule_segment\": {}, \"row_ref_base\": {}, \"bucket_keys\": {}, \"seed_step\": {}, \"work_sections\": {}}}}},\n",
        wgsl::meta::LEN, wgsl::meta::BASE, wgsl::meta::SEG, wgsl::meta::REF, wgsl::meta::KEYS, wgsl::meta::SEED, wgsl::meta::W_COUNTS
    );
    let order: Vec<&str> = Shape::ALL.iter().map(|x| x.name()).collect();
    s += &format!(
        "    \"cols\": {{\"binding\": 1, \"access\": \"read\", \"layout\": \"per shape in table order {}; per shape its columns, each table_len words\"}},\n",
        order.join(",")
    );
    s += "    \"work\": {\"binding\": 2, \"access\": \"read_write atomic\", \"sections\": [\"counts\", \"start\", \"bucket_rows\", \"count\", \"offset\", \"best(hi,lo,idx)\", \"taken\", \"ctrl\"]},\n";
    s += &format!("    \"redexes\": {{\"binding\": 3, \"access\": \"read_write\", \"words_per_redex\": {}, \"fields\": [\"rule\", \"consumer\", \"p0\", \"p1\", \"p2\", \"prio_hi\", \"prio_lo\", \"decision\"]}},\n", wgsl::REDEX_WORDS);
    s += "    \"params\": {\"binding\": 4, \"uniform\": true, \"dynamic_offset_stride\": 256}\n  },\n";
    s += "  \"step\": [\"k_fill(counts)\", \"k_hist\", \"k_scan(counts->start)\", \"k_scatter\", \"k_bsort\", \"k_count\", \"k_scan(count->offset)\", \"read total\", \"k_emit\", \"repeat {k_fill(best,taken,ctrl), k_min_hi, k_min_lo, k_min_idx, k_take, k_exclude, read ctrl} until 0\", \"read redexes; host commit\"],\n";
    s += &format!(
        "  \"reference\": {{\"machine\": \"host\", \"seed\": {}, \"resolver\": \"{}\", \"max_steps\": {}, \"stop\": \"{}\", \"steps\": {}, \"fired\": {}, \"final_hash\": \"{}\"}}\n",
        cfg.seed,
        cfg.resolver.name(),
        cfg.max_steps,
        esc(&format!("{:?}", reference.stop)),
        reference.steps.len(),
        reference.fired,
        hex(&reference.final_hash())
    );
    s += "}\n";
    s
}

/// Write the GPU target bundle for `term` into `dir`.
pub fn write_bundle(dir: &Path, term: &Term, source_hash: Option<&Hash32>, cfg: &Config) -> std::io::Result<RunReport> {
    device_accepts(term).map_err(std::io::Error::other)?;
    std::fs::create_dir_all(dir)?;
    let mut it = InternTable::default();
    let st = State::import(term, &mut it).map_err(|e| std::io::Error::other(e.to_string()))?;
    let cmat = cmat_encode(&st, &mut it);
    let kernels = wgsl::kernels();
    let reference = f1r3comb_par::run(term, cfg, &mut Host);
    std::fs::write(dir.join("program.cmat"), &cmat)?;
    std::fs::write(dir.join("kernels.wgsl"), &kernels)?;
    std::fs::write(dir.join("manifest.json"), manifest(term, source_hash, &cmat, &kernels, &reference, cfg))?;
    Ok(reference)
}

/// Read a bundle's program back as a term (and check the kernels are the
/// ones this build generates).
pub fn read_bundle(dir: &Path) -> Result<(Term, bool), String> {
    let cmat = std::fs::read(dir.join("program.cmat")).map_err(|e| format!("program.cmat: {e}"))?;
    let (st, mut it) = cmat_decode(&cmat, NameBudget::default()).map_err(|e| e.to_string())?;
    let kernels_match = std::fs::read_to_string(dir.join("kernels.wgsl")).map(|k| k == wgsl::kernels()).unwrap_or(false);
    Ok((st.export(&mut it), kernels_match))
}

#[cfg(test)]
mod tests;
