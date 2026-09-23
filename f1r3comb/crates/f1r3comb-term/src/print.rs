//! The textual form of the target: a human-readable rendering with a name
//! table. The canonical artefact is the binary encoding.
//!
//! Name labels: `@0`; `K`; `r*` and `f*`, the static channels; `□k` for the
//! hole of context level `k`; `ρ#hhhhhhhh` for a root address; `x.0110` for
//! the leaf at path `0110` beneath `x`; and `#hhhhhhhh` for every other name,
//! whose definition is listed once in the name table.

use crate::addr::{f_star, k_const, r_star, spine};
use crate::{Atom, CName, Shape, Term};
use std::collections::HashSet;
use std::fmt::Write;

fn bits(w: &[bool]) -> String {
    w.iter().map(|b| if *b { '1' } else { '0' }).collect()
}

fn base_label(n: &CName) -> String {
    if let Some(k) = n.is_hole() {
        return format!("□{k}");
    }
    if n.drop().is_nil() {
        return "@0".into();
    }
    if *n == k_const() {
        return "K".into();
    }
    if *n == r_star() {
        return "r*".into();
    }
    if *n == f_star() {
        return "f*".into();
    }
    if crate::addr::is_root(n) {
        return format!("ρ#{}", &n.content_hash().hex()[..8]);
    }
    format!("#{}", &n.content_hash().hex()[..8])
}

pub fn name_label(n: &CName) -> String {
    let (b, w) = spine(n);
    if !w.is_empty() && (b.is_hole().is_some() || crate::addr::is_root(&b)) {
        return format!("{}.{}", base_label(&b), bits(&w));
    }
    base_label(n)
}

fn is_listed(n: &CName) -> bool {
    name_label(n).starts_with('#')
}

pub fn atom_inline(a: &Atom) -> String {
    let mut s = String::new();
    s.push_str(a.shape().name());
    s.push('(');
    let parts: Vec<String> = a.names().iter().map(name_label).collect();
    s.push_str(&parts.join(", "));
    if let Some(p) = a.store() {
        s.push_str(", ");
        s.push_str(&term_inline(p));
    }
    if let Some(c) = a.context() {
        s.push_str(", ctx{");
        s.push_str(&term_inline(c));
        s.push('}');
    }
    s.push(')');
    s
}

pub fn term_inline(t: &Term) -> String {
    if t.is_nil() {
        return "0".to_string();
    }
    let parts: Vec<String> = t.atoms().iter().map(atom_inline).collect();
    parts.join(" | ")
}

fn term_block(t: &Term, indent: &str, out: &mut String) {
    if t.is_nil() {
        let _ = writeln!(out, "{indent}0");
        return;
    }
    for a in t.atoms() {
        let names: Vec<String> = a.names().iter().map(name_label).collect();
        match a.shape() {
            Shape::Q => {
                let _ = writeln!(out, "{indent}q({}, {{", names[0]);
                term_block(a.store().unwrap(), &format!("{indent}    "), out);
                let _ = writeln!(out, "{indent}}})");
            }
            Shape::Inst | Shape::Cstar => {
                let _ = writeln!(out, "{indent}{}({}, ctx {{", a.shape().name(), names.join(", "));
                term_block(a.context().unwrap(), &format!("{indent}    "), out);
                let _ = writeln!(out, "{indent}}})");
            }
            _ => {
                let _ = writeln!(out, "{indent}{}", atom_inline(a));
            }
        }
    }
}

fn listed_names(t: &Term) -> Vec<CName> {
    let mut seen: HashSet<Vec<u8>> = HashSet::new();
    let mut order = Vec::new();
    let mut work: Vec<Term> = vec![t.clone()];
    while let Some(t) = work.pop() {
        for a in t.atoms().iter().rev() {
            if let Some(p) = a.store().or(a.context()) {
                work.push(p.clone());
            }
            for n in a.names().iter().rev() {
                if seen.insert(n.encode().to_vec()) {
                    if is_listed(n) {
                        order.push(n.clone());
                        work.push(n.drop());
                    }
                }
            }
        }
    }
    order
}

/// The full textual form of a compiled term.
pub fn render(t: &Term, header: &[(&str, String)]) -> String {
    let mut out = String::new();
    let _ = writeln!(out, "; F1R3Comb target (rho combinators, presentation A)");
    for (k, v) in header {
        let _ = writeln!(out, "; {k}: {v}");
    }
    let _ = writeln!(out, "; term hash: {}", t.content_hash().hex());
    let _ = writeln!(out, "; atoms: {}", t.len());
    let _ = writeln!(out, "term {{");
    term_block(t, "    ", &mut out);
    let _ = writeln!(out, "}}");
    let names = listed_names(t);
    if !names.is_empty() {
        let _ = writeln!(out);
        let _ = writeln!(out, "names {{");
        for n in names {
            let _ = writeln!(out, "    {} = @{{", name_label(&n));
            term_block(&n.drop(), "        ", &mut out);
            let _ = writeln!(out, "    }}");
        }
        let _ = writeln!(out, "}}");
    }
    out
}
