//! `f1r3comb` — the staged compiler driver.
//!
//! ```text
//! f1r3comb compile PROGRAM.rho [-o OUT.comb] [--emit-ir-text OUT.txt] [--emit-gpu DIR] [--scheme inst|curried|positional]
//! f1r3comb run INPUT [--machine host|gpu] [--seed N] [--resolver R] [--max-steps N] [--trace] [--out FINAL.comb]
//! f1r3comb print INPUT            textual IR
//! f1r3comb lattice INPUT          expressiveness-lattice audit
//! f1r3comb hash INPUT             term content hash
//! ```
//!
//! INPUT is any stage's artefact: K0 source text, a saved `.comb` IR, a
//! `.cmat` matrix file, or a GPU bundle directory.

use f1r3comb_mat::{cmat_decode, NameBudget};
use f1r3comb_par::{Config, Resolver};
use f1r3comb_term::artefact::{Artefact, Header, Scheme};
use f1r3comb_term::addr::Allocator;
use f1r3comb_term::{Hash32, Term};
use std::path::{Path, PathBuf};
use std::process::ExitCode;

const USAGE: &str = "usage:
  f1r3comb compile PROGRAM.rho [-o OUT.comb] [--emit-ir-text OUT.txt] [--emit-gpu DIR] [--seed N] [--max-steps N]
                   [--scheme inst|curried|positional]   (curried: the logic's build; positional: the unsound control)
  f1r3comb run INPUT [--machine host|gpu] [--seed N] [--resolver maximal-progress|single-by-priority|single-uniform|cross-instance]
                     [--max-steps N] [--trace] [--out FINAL.comb]
  f1r3comb print INPUT
  f1r3comb lattice INPUT
  f1r3comb hash INPUT
INPUT: K0 source, .comb (saved IR), .cmat, or a GPU bundle directory";

struct Args {
    pos: Vec<String>,
    opts: Vec<(String, Option<String>)>,
}

impl Args {
    fn parse(v: Vec<String>) -> Args {
        let flags = ["--trace"];
        let mut pos = Vec::new();
        let mut opts = Vec::new();
        let mut it = v.into_iter();
        while let Some(a) = it.next() {
            if a.starts_with('-') {
                if flags.contains(&a.as_str()) {
                    opts.push((a, None));
                } else {
                    let val = it.next();
                    opts.push((a, val));
                }
            } else {
                pos.push(a);
            }
        }
        Args { pos, opts }
    }
    fn get(&self, k: &str) -> Option<&str> {
        self.opts.iter().rev().find(|(n, _)| n == k).and_then(|(_, v)| v.as_deref())
    }
    fn has(&self, k: &str) -> bool {
        self.opts.iter().any(|(n, _)| n == k)
    }
    fn num(&self, k: &str, d: u64) -> Result<u64, String> {
        match self.get(k) {
            None => Ok(d),
            Some(s) => s.parse().map_err(|_| format!("{k}: not a number: {s}")),
        }
    }
}

struct Loaded {
    term: Term,
    header: Option<Header>,
    /// The phase-one image, when compiled from source.
    ir: Option<f1r3comb_ir::Node>,
    /// A bundle or matrix state is already deployed; a compiled term is not.
    deployed: bool,
    kind: &'static str,
}

fn scheme(a: &Args) -> Result<Scheme, String> {
    match a.get("--scheme").unwrap_or("inst") {
        "inst" => Ok(Scheme::Inst),
        "positional" => Ok(Scheme::Positional),
        "curried" => Ok(Scheme::Curried),
        s => Err(format!("unknown scheme {s}")),
    }
}

fn load(path: &str, a: &Args) -> Result<Loaded, String> {
    let p = Path::new(path);
    if p.is_dir() {
        let (term, same) = f1r3comb_gpu::read_bundle(p)?;
        if !same {
            eprintln!("warning: {path}/kernels.wgsl differs from this build's kernels; running this build's");
        }
        return Ok(Loaded { term, header: None, ir: None, deployed: true, kind: "bundle" });
    }
    let bytes = std::fs::read(p).map_err(|e| format!("{path}: {e}"))?;
    if Artefact::is_artefact(&bytes) {
        let art = Artefact::decode(&bytes).map_err(|e| e.to_string())?;
        return Ok(Loaded { term: art.term, header: Some(art.header), ir: None, deployed: false, kind: "comb" });
    }
    if bytes.starts_with(f1r3comb_mat::CMAT_MAGIC) {
        let (st, mut it) = cmat_decode(&bytes, NameBudget::default()).map_err(|e| e.to_string())?;
        return Ok(Loaded { term: st.export(&mut it), header: None, ir: None, deployed: true, kind: "cmat" });
    }
    let src = String::from_utf8(bytes).map_err(|_| format!("{path}: not UTF-8 source and not a known artefact"))?;
    match f1r3comb_lower::compile_source_with(&src, scheme(a)?) {
        Ok(c) => {
            for w in &c.warnings {
                eprintln!("{path}: {w}");
            }
            Ok(Loaded { term: c.term, header: Some(c.header), ir: Some(c.ir), deployed: false, kind: "source" })
        }
        Err(ds) => Err(ds.iter().map(|d| format!("{path}: {d}")).collect::<Vec<_>>().join("\n")),
    }
}

fn hex(h: &Hash32) -> String {
    h.0.iter().map(|b| format!("{b:02x}")).collect()
}

fn family_names(f: u16) -> String {
    let mut v: Vec<String> = f1r3comb_term::Shape::BASES
        .iter()
        .filter(|s| f & (1 << s.ix()) != 0)
        .map(|s| format!("cons_{}", s.name()))
        .collect();
    if f & (1 << 15) != 0 {
        v.insert(0, "cons_|".into());
    }
    format!("{{{}}}", v.join(", "))
}

fn header_lines(l: &Loaded) -> Vec<(&'static str, String)> {
    let mut hdr = vec![("input", l.kind.to_string())];
    if let Some(h) = &l.header {
        hdr.push(("erection", h.scheme.name().to_string()));
        if h.source_hash.0 != [0; 32] {
            hdr.push(("source hash", hex(&h.source_hash)));
        }
        hdr.push(("family used, F(P)", family_names(h.family)));
        hdr.push(("marking", format!("{} receives, {} unsafe for the fixed-arity family", h.units, h.unsafe_units)));
        hdr.push(("root address must exceed", format!("{} bytes", h.size_bound)));
    }
    hdr
}

fn target_text(l: &Loaded) -> String {
    f1r3comb_term::print::render(&l.term, &header_lines(l))
}

fn ir_text(l: &Loaded) -> String {
    let mut s = String::new();
    if let Some(ir) = &l.ir {
        s += "; phase one: RC^nu (the combinators with a binder)\n";
        s += &format!("; ir hash: {}\n", hex(&ir.hash()));
        s += &ir.render();
        s += "\n";
    }
    s += "; phase two:\n";
    s += &target_text(l);
    s
}

fn config(a: &Args) -> Result<Config, String> {
    let resolver = match a.get("--resolver") {
        None => Resolver::MaximalProgress,
        Some(s) => Resolver::parse(s).ok_or(format!("unknown resolver {s}"))?,
    };
    Ok(Config {
        seed: a.num("--seed", 0)?,
        max_steps: a.num("--max-steps", 100_000)?,
        resolver,
        hash_states: a.has("--trace"),
        ..Default::default()
    })
}

/// The deployed state: the term plus a fresh root address at `r*`.
fn deployed(l: &Loaded) -> Result<Term, String> {
    if l.deployed {
        return Ok(l.term.clone());
    }
    if let Some(h) = &l.header {
        if h.scheme == Scheme::Positional {
            eprintln!("warning: deploying a positional (negative-control) artefact; it is unsound on programs that run received code twice");
        }
    }
    let (t, root) = f1r3comb_par::deploy(&l.term, &mut Allocator::new());
    if let Some(r) = root {
        eprintln!("deploy: root address {} at r* ({} bytes)", f1r3comb_term::print::name_label(&r), r.size());
    }
    Ok(t)
}

fn compile(a: &Args) -> Result<(), String> {
    let input = a.pos.get(1).ok_or(USAGE)?;
    let l = load(input, a)?;
    let stem = PathBuf::from(input).with_extension("");
    let out = a.get("-o").map(PathBuf::from).unwrap_or_else(|| stem.with_extension("comb"));
    let header = l.header.clone().ok_or("compile takes K0 source")?;
    let art = Artefact { header: header.clone(), term: l.term.clone() };
    std::fs::write(&out, art.encode()).map_err(|e| format!("{}: {e}", out.display()))?;
    eprintln!(
        "wrote {} ({} erection, {} top-level atoms, term {}, marking {}/{} unsafe)",
        out.display(),
        header.scheme.name(),
        l.term.len(),
        &hex(&l.term.content_hash())[..16],
        header.unsafe_units,
        header.units
    );
    if let Some(t) = a.get("--emit-ir-text") {
        std::fs::write(t, ir_text(&l)).map_err(|e| format!("{t}: {e}"))?;
        eprintln!("wrote IR text {t} (phase one and phase two)");
    }
    if let Some(d) = a.get("--emit-gpu") {
        let cfg = config(a)?;
        let t = deployed(&l)?;
        let r = f1r3comb_gpu::write_bundle(Path::new(d), &t, Some(&header.source_hash), &cfg).map_err(|e| format!("{d}: {e}"))?;
        eprintln!(
            "wrote GPU bundle {d}/ (program.cmat, kernels.wgsl, manifest.json); host reference: {} steps, {} firings, {:?}",
            r.steps.len(),
            r.fired,
            r.stop
        );
    }
    Ok(())
}

fn run(a: &Args) -> Result<(), String> {
    let input = a.pos.get(1).ok_or(USAGE)?;
    let l = load(input, a)?;
    let cfg = config(a)?;
    let t = deployed(&l)?;
    let machine = a.get("--machine").unwrap_or("host");
    let report = match machine {
        "host" => f1r3comb_par::run(&t, &cfg, &mut f1r3comb_par::Host),
        "gpu" => {
            f1r3comb_gpu::device_accepts(&t)?;
            if cfg.resolver == Resolver::CrossInstance {
                return Err("the cross-instance resolver needs the intern table and runs on the host only".into());
            }
            run_gpu(&t, &cfg)?
        }
        m => return Err(format!("unknown machine {m}")),
    };
    println!("machine {machine}  seed {}  resolver {}", cfg.seed, cfg.resolver.name());
    if a.has("--trace") {
        println!("step  enumerated  fired  names  per-rule                                  state");
        for s in &report.steps {
            let rules: Vec<String> = s
                .per_rule
                .iter()
                .enumerate()
                .filter(|(_, n)| **n > 0)
                .map(|(i, n)| format!("{}:{n}", f1r3comb_term::rules::rules()[i].name))
                .collect();
            let h = s.state_hash.map(|h| hex(&h)[..16].to_string()).unwrap_or_default();
            println!("{:4}  {:10}  {:5}  {:5}  {:40}  {h}", s.step, s.enumerated, s.fired, s.names_built, rules.join(" "));
        }
    }
    let built: u64 = report.steps.iter().map(|s| s.names_built).sum();
    println!(
        "stop {:?}  steps {}  fired {}  names built {}  final atoms {}  final hash {}",
        report.stop,
        report.steps.len(),
        report.fired,
        built,
        report.final_term.len(),
        hex(&report.final_hash())
    );
    if let Some(o) = a.get("--out") {
        let h = Header {
            scheme: l.header.as_ref().map(|h| h.scheme).unwrap_or(Scheme::Inst),
            source_hash: Hash32([0; 32]),
            family: 0,
            units: 0,
            unsafe_units: 0,
            size_bound: 0,
        };
        let art = Artefact { header: h, term: report.final_term.clone() };
        std::fs::write(o, art.encode()).map_err(|e| format!("{o}: {e}"))?;
    }
    Ok(())
}

fn lattice(l: &Loaded) -> String {
    let mut s = String::new();
    if let Some(ir) = &l.ir {
        let (top, her) = ir.counts();
        s += &f1r3comb_term::lattice::report_counts("phase-one image (RC^nu) -- the informative point", &top, &her);
        s += "\n";
    }
    let t = &l.term;
    s += &f1r3comb_term::lattice::report_counts(
        "phase-two image (target)",
        &f1r3comb_term::lattice::top_level(t),
        &f1r3comb_term::lattice::hereditary(t),
    );
    if l.header.as_ref().map(|h| h.scheme) == Some(Scheme::Inst) {
        s += "  note: under the inst erection each unit's top level is one inst shape (F1R3Comb v0.6 Req. 10.3);\n";
        s += "        compile with --scheme curried for a per-shape phase-two report (where the curried erection reaches).\n";
    }
    s
}

#[cfg(feature = "gpu")]
fn run_gpu(t: &Term, cfg: &Config) -> Result<f1r3comb_par::RunReport, String> {
    let mut g = f1r3comb_gpu::device::Gpu::new().map_err(|e| e.to_string())?;
    eprintln!("device: {} ({})", g.adapter_name, g.backend);
    let r = f1r3comb_par::run(t, cfg, &mut g);
    eprintln!("device stats: {:?}", g.stats);
    Ok(r)
}

#[cfg(not(feature = "gpu"))]
fn run_gpu(_: &Term, _: &Config) -> Result<f1r3comb_par::RunReport, String> {
    Err("this build has no GPU backend (build with feature `gpu`)".into())
}

fn main() -> ExitCode {
    let a = Args::parse(std::env::args().skip(1).collect());
    let arg = |a: &Args| a.pos.get(1).cloned().ok_or(USAGE.to_string());
    let res = match a.pos.first().map(|s| s.as_str()) {
        Some("compile") => compile(&a),
        Some("run") => run(&a),
        Some("print") => arg(&a).and_then(|p| load(&p, &a)).map(|l| print!("{}", ir_text(&l))),
        Some("lattice") => arg(&a).and_then(|p| load(&p, &a)).map(|l| print!("{}", lattice(&l))),
        Some("hash") => arg(&a).and_then(|p| load(&p, &a)).map(|l| println!("{}", hex(&l.term.content_hash()))),
        _ => Err(USAGE.to_string()),
    };
    match res {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("{e}");
            ExitCode::FAILURE
        }
    }
}
