//! `f1r3comb-lower` — the two-phase translation of draft 3 §6 (F1R3Comb v0.6
//! §5, §8): K0 normal form → RC^ν (phase one, [`phase1`]) → the rho
//! combinators with `inst` erection (phase two, [`phase2`]), with the
//! post-passes of [`verify`] and the marking analysis of Req. 5.17.
//!
//! Canonicity (Obl. 6.2): phase one is a homomorphism on the normal form
//! with no allocation; phase two introduces only the two global static
//! channels and leaves written beneath holes. The artefact contains no
//! instance name at all; freshness happens at deploy, by the address.

#![forbid(unsafe_code)]

pub mod check;
pub mod phase1;
pub mod phase2;
pub mod verify;

use f1r3comb_ir::{IName, Node};
use f1r3comb_term::artefact::{Header, Scheme};
use f1r3comb_term::{Hash32, Shape, Term};
use k1ndl1ng_norm::Norm;
use std::collections::HashSet;

pub use phase2::is_apparatus;

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum Severity {
    Error,
    Warning,
}

#[derive(Clone, Debug)]
pub struct Diag {
    pub code: &'static str,
    pub severity: Severity,
    pub span: Option<(u32, u32)>,
    pub message: String,
}

impl std::fmt::Display for Diag {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let sev = match self.severity {
            Severity::Error => "error",
            Severity::Warning => "warning",
        };
        match self.span {
            Some((lo, hi)) => write!(f, "{sev}[{}] {lo}..{hi}: {}", self.code, self.message),
            None => write!(f, "{sev}[{}]: {}", self.code, self.message),
        }
    }
}

/// The result of compiling a source program.
#[derive(Clone)]
pub struct Compiled {
    pub source: Norm,
    pub source_hash: Hash32,
    /// The phase-one image.
    pub ir: Node,
    /// The phase-two image.
    pub term: Term,
    pub header: Header,
    pub warnings: Vec<Diag>,
}

/// Compile K0 source text under the default `inst` erection.
pub fn compile_source(src: &str) -> Result<Compiled, Vec<Diag>> {
    compile_source_with(src, Scheme::Inst)
}

pub fn compile_source_with(src: &str, scheme: Scheme) -> Result<Compiled, Vec<Diag>> {
    let opts = k1ndl1ng_parse::Options { level: k1ndl1ng_parse::Level::K0, ..Default::default() };
    let parsed = k1ndl1ng_parse::parse(src, &opts);
    let mut diags = Vec::new();
    for d in &parsed.diags {
        let code = if d.code == "level" { "comb-level" } else { d.code };
        diags.push(Diag {
            code,
            severity: match d.severity {
                k1ndl1ng_parse::Severity::Error => Severity::Error,
                k1ndl1ng_parse::Severity::Warning => Severity::Warning,
            },
            span: Some((d.span.lo, d.span.hi)),
            message: d.message.clone(),
        });
    }
    if !parsed.ok() {
        return Err(diags);
    }
    check::guard_check(&parsed.tree, &mut diags);
    let nopts = k1ndl1ng_norm::Options { require_closed: true, ..Default::default() };
    let normal = match k1ndl1ng_norm::normalise(&parsed.tree, &nopts) {
        Ok(n) => n,
        Err(ds) => {
            for d in ds {
                let code = match d.code {
                    c if c.contains("free") || c.contains("open") || c.contains("closed") => "comb-open",
                    c => c,
                };
                diags.push(Diag { code, severity: Severity::Error, span: Some((d.span.lo, d.span.hi)), message: d.message });
            }
            return Err(diags);
        }
    };
    match compile_norm_with(&normal.term, scheme) {
        Ok(mut c) => {
            diags.append(&mut c.warnings);
            c.warnings = diags;
            Ok(c)
        }
        Err(mut e) => {
            diags.append(&mut e);
            Err(diags)
        }
    }
}

pub fn compile_norm(p: &Norm) -> Result<Compiled, Vec<Diag>> {
    compile_norm_with(p, Scheme::Inst)
}

/// Compile a closed K0 normal form.
pub fn compile_norm_with(p: &Norm, scheme: Scheme) -> Result<Compiled, Vec<Diag>> {
    let mut diags = Vec::new();
    check::level_and_closed(p, &mut diags);
    if scheme == Scheme::Curried {
        diags.push(Diag {
            code: "comb-scheme-mismatch",
            severity: Severity::Error,
            span: None,
            message: "this build does not emit the curried erection (feature curried-erection is not implemented); \
                      the machine runs curried templates, see f1r3comb-check's E5-E7"
                .into(),
        });
    }
    if diags.iter().any(|d| d.severity == Severity::Error) {
        return Err(diags);
    }
    // phase one
    let mut p1 = phase1::Phase1::default();
    let ir = p1.closed(p);
    diags.append(&mut p1.diags);
    if diags.iter().any(|d| d.severity == Severity::Error) {
        return Err(diags);
    }
    // phase two
    let mut p2 = phase2::Phase2::new(scheme);
    let term = p2.closed(&ir);
    // post-passes
    if scheme == Scheme::Inst {
        verify::check(&term, &mut diags);
    }
    if diags.iter().any(|d| d.severity == Severity::Error) {
        return Err(diags);
    }
    let (receives, unsafe_receives) = marking(&ir, &p1.marks);
    let header = Header {
        scheme,
        source_hash: p.hash(),
        family: family(&term),
        units: receives,
        unsafe_units: unsafe_receives,
        size_bound: f1r3comb_term::addr::largest_name(&term).size() as u64,
    };
    Ok(Compiled { source: p.clone(), source_hash: p.hash(), ir, term, header, warnings: diags })
}

/// `[[@P]]` for a closed normal form `P` (the name a source name compiles to).
pub fn compile_name(p: &Norm) -> Result<f1r3comb_term::CName, Vec<Diag>> {
    compile_norm(p).map(|c| f1r3comb_term::CName::quote(c.term))
}

/// F(P) as used: bit `i` for `cons_A` of base shape `i`, bit 15 for `cons_|`.
fn family(t: &Term) -> u16 {
    let mut f = 0u16;
    for a in verify::all_atoms(t) {
        if let Some(b) = a.shape().builds() {
            f |= 1 << b.ix();
        }
        if a.shape() == Shape::ConsPar {
            f |= 1 << 15;
        }
    }
    f
}

/// The marking analysis of draft 3 Rem. 6.20 (Req. 5.17): a receive is
/// unsafe for the fixed-arity family iff some atom of its phase-one image
/// mentions marked names of two different flows. Returns (receives, unsafe).
pub fn marking(ir: &Node, marks: &phase1::Marks) -> (u64, u64) {
    let mut unsafe_: HashSet<u32> = HashSet::new();
    let mut work = vec![ir];
    while let Some(n) = work.pop() {
        match n {
            Node::Nil => {}
            Node::Par(xs) => work.extend(xs.iter()),
            Node::New { body, .. } => work.push(body),
            Node::Atom { shape, names, store } => {
                let mut groups: Vec<(u32, u32)> = Vec::new();
                // a distributor splits one flow into many; it is where the
                // correlation is created, not where two are joined
                let splits = *shape == Shape::D;
                for x in names {
                    match x {
                        IName::Var(v) => {
                            if let Some(g) = marks.group.get(v) {
                                groups.push(*g);
                            }
                        }
                        IName::Quote(q) => work.push(q),
                    }
                }
                groups.sort_unstable();
                groups.dedup();
                if groups.len() > 1 && !splits {
                    unsafe_.extend(groups.iter().map(|g| g.0));
                }
                if let Some(s) = store {
                    work.push(s);
                }
            }
        }
    }
    (marks.receives as u64, unsafe_.len() as u64)
}

#[cfg(test)]
mod tests;
