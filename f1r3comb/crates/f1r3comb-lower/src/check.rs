//! Phases 1, 2 and 4 of the lowering (F1R3Comb v0.6 §8.1): the level check,
//! the guarded-drop check on the tree, and the closedness check on the normal
//! form.

use crate::{Diag, Severity};
use k1ndl1ng_ast::{Ast, NameNode, ProcId, ProcNode};
use k1ndl1ng_norm::{BindKind, Name, Node, Norm};

/// Req. 3.4 under presentation A: a literal `*@Q` is collapsed by the
/// equation on both sides of the compiler, so it is a warning. (Unguarded
/// drops of unbound names are free names, which K0 closedness rejects as
/// `comb-open`.)
pub fn guard_check(ast: &Ast, out: &mut Vec<Diag>) {
    for i in 0..ast.proc_count() {
        let p = ProcId(i);
        if let ProcNode::Eval { name } = ast.proc(p) {
            if let NameNode::Quote(_) = ast.name(name) {
                let sp = ast.proc_span(p);
                out.push(Diag {
                    code: "comb-drop-quote",
                    severity: Severity::Warning,
                    span: Some((sp.lo, sp.hi)),
                    message: "`*@Q` is collapsed to `Q` by the equation *@p = p, which both \
                              the source normaliser and presentation A satisfy"
                        .into(),
                });
            }
        }
    }
}

/// K0 and closedness on the normal form (Req. 3.1, 3.2). Every independent
/// error is reported.
pub fn level_and_closed(t: &Norm, out: &mut Vec<Diag>) {
    let mut work: Vec<&Norm> = vec![t];
    let mut names: Vec<&Name> = Vec::new();
    let err = |out: &mut Vec<Diag>, code: &'static str, msg: String| {
        out.push(Diag { code, severity: Severity::Error, span: None, message: msg })
    };
    while let Some(p) = work.pop() {
        match p.node() {
            Node::Nil => {}
            Node::Par(ps) => work.extend(ps.iter()),
            Node::Send { chan, persistent, args } => {
                if *persistent {
                    err(out, "comb-level", "persistent send `!!` is K1; the combinators have no persistent atom".into());
                }
                if args.len() != 1 {
                    err(out, "comb-level", format!("a send with {} arguments is K1; K0 sends are monadic", args.len()));
                }
                names.push(chan);
                work.extend(args.iter());
            }
            Node::Receive { binds, body } => {
                if binds.len() != 1 {
                    err(out, "comb-level", "a join is K1; K0 receives have one bind".into());
                }
                for b in binds {
                    if b.kind != BindKind::Linear {
                        err(out, "comb-level", "persistent receive or peek is K1".into());
                    }
                    if b.pats.len() != 1 {
                        err(out, "comb-level", "a polyadic receipt is K1".into());
                    }
                    for pat in &b.pats {
                        if !matches!(pat, Name::Free(_)) {
                            err(out, "comb-level", "a structured pattern is K1; K0 binds a name variable".into());
                        }
                    }
                    names.push(&b.chan);
                }
                work.push(body);
            }
            Node::New { .. } => {
                err(out, "comb-level", "`new` is K1; ground names by quotation".into())
            }
            Node::Eval(n) => names.push(n),
            Node::BoundVar(_) | Node::FreeVar(_) | Node::Wild => {
                err(out, "comb-level", "process variables and wildcards are K1".into())
            }
        }
        while let Some(n) = names.pop() {
            match n {
                Name::Quote(q) => work.push(q),
                Name::Bound(_) => {}
                Name::Free(_) => err(
                    out,
                    "comb-open",
                    "a free name survived: the input must be closed, grounded by quotation".into(),
                ),
                Name::Unforgeable(_) => err(
                    out,
                    "comb-unforgeable",
                    "unforgeable names have no target former at v1; ground by quotation".into(),
                ),
            }
        }
    }
}
