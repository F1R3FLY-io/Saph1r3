//! `unnormalise`: a normal form back to a tree, with generated identifiers in
//! place of indices, so a term arriving from the store or the wire can be
//! printed and shown (spec §3.2). Iterative, like everything else.

use crate::build::Sort;
use crate::term::*;
use k1ndl1ng_ast::print::{print_proc, Style};
use k1ndl1ng_ast::{Ast, BindKind, Builder, NameId, ProcId};

/// Sorts of a bind's binder slots, read off its patterns.
pub fn slot_sorts(pats: &[Name], binders: u16) -> Vec<Sort> {
    let mut sorts = vec![Sort::Name; binders as usize];
    let mut names: Vec<Name> = pats.to_vec();
    let mut procs: Vec<Norm> = Vec::new();
    loop {
        while let Some(n) = names.pop() {
            match n {
                Name::Free(l) => {
                    if (l as usize) < sorts.len() {
                        sorts[l as usize] = Sort::Name;
                    }
                }
                Name::Quote(q) => procs.push(q),
                _ => {}
            }
        }
        let p = match procs.pop() {
            Some(p) => p,
            None => break,
        };
        match p.node() {
            Node::FreeVar(l) => {
                if (*l as usize) < sorts.len() {
                    sorts[*l as usize] = Sort::Proc;
                }
            }
            Node::Par(v) => procs.extend(v.iter().cloned()),
            Node::Send { chan, args, .. } => {
                names.push(chan.clone());
                procs.extend(args.iter().cloned());
            }
            Node::Receive { binds, body } => {
                for b in binds {
                    names.push(b.chan.clone());
                    for q in &b.pats {
                        names.push(q.clone());
                    }
                }
                procs.push(body.clone());
            }
            Node::New { body, .. } => procs.push(body.clone()),
            Node::Eval(n) => names.push(n.clone()),
            _ => {}
        }
    }
    sorts
}

enum U {
    P(Norm),
    N(Name),
    MkPar(u32),
    MkSend(bool, u32),
    MkEval,
    MkQuote,
    MkNew(u32),
    MkRecv(u32, u32),
    MkBind(BindKind, u32),
    PushPat(Vec<String>),
    PopPat,
    PushBinders(Vec<String>),
}

struct Gen {
    n: u32,
}

impl Gen {
    fn next(&mut self, sort: Sort) -> String {
        let s = match sort {
            Sort::Name => format!("x{}", self.n),
            Sort::Proc => format!("P{}", self.n),
        };
        self.n += 1;
        s
    }
}

/// A normal form as a tree, ready to print.
pub fn unnormalise(t: &Norm) -> Ast {
    let mut b = Builder::new();
    let mut gen = Gen { n: 0 };
    let mut binders: Vec<String> = Vec::new();
    let mut patframes: Vec<Vec<String>> = Vec::new();
    let mut procs: Vec<ProcId> = Vec::new();
    let mut names: Vec<NameId> = Vec::new();
    let mut binds: Vec<(BindKind, Vec<NameId>, NameId)> = Vec::new();
    let mut jobs: Vec<U> = vec![U::P(t.clone())];

    while let Some(j) = jobs.pop() {
        match j {
            U::P(p) => match p.node() {
                Node::Nil => {
                    let x = b.nil();
                    procs.push(x);
                }
                Node::Wild => {
                    let x = b.add_proc(k1ndl1ng_ast::ProcNode::Wild, k1ndl1ng_ast::Span::NULL);
                    procs.push(x);
                }
                Node::BoundVar(i) => {
                    let id = binder_name(&binders, *i);
                    let x = b.proc_var(&id);
                    procs.push(x);
                }
                Node::FreeVar(l) => {
                    let id = match patframes.last() {
                        Some(f) => f
                            .get(*l as usize)
                            .cloned()
                            .unwrap_or_else(|| format!("P{l}")),
                        None => format!("f{l}"),
                    };
                    let x = b.proc_var(&id);
                    procs.push(x);
                }
                Node::Par(v) => {
                    jobs.push(U::MkPar(v.len() as u32));
                    for c in v.iter().rev() {
                        jobs.push(U::P(c.clone()));
                    }
                }
                Node::Send {
                    chan,
                    persistent,
                    args,
                } => {
                    jobs.push(U::MkSend(*persistent, args.len() as u32));
                    for a in args.iter().rev() {
                        jobs.push(U::P(a.clone()));
                    }
                    jobs.push(U::N(chan.clone()));
                }
                Node::Eval(n) => {
                    jobs.push(U::MkEval);
                    jobs.push(U::N(n.clone()));
                }
                Node::New { count, body } => {
                    let ids: Vec<String> = (0..*count).map(|_| gen.next(Sort::Name)).collect();
                    jobs.push(U::MkNew(*count));
                    jobs.push(U::P(body.clone()));
                    jobs.push(U::PushBinders(ids));
                }
                Node::Receive { binds: bs, body } => {
                    let mut all: Vec<String> = Vec::new();
                    let mut per_bind: Vec<Vec<String>> = Vec::new();
                    for bd in bs {
                        let sorts = slot_sorts(&bd.pats, bd.binders);
                        let ids: Vec<String> = sorts.iter().map(|s| gen.next(*s)).collect();
                        all.extend(ids.iter().cloned());
                        per_bind.push(ids);
                    }
                    jobs.push(U::MkRecv(bs.len() as u32, all.len() as u32));
                    jobs.push(U::P(body.clone()));
                    jobs.push(U::PushBinders(all));
                    for (bd, ids) in bs.iter().zip(per_bind).rev() {
                        jobs.push(U::MkBind(bd.kind, bd.pats.len() as u32));
                        jobs.push(U::PopPat);
                        for p in bd.pats.iter().rev() {
                            jobs.push(U::N(p.clone()));
                        }
                        jobs.push(U::PushPat(ids));
                        jobs.push(U::N(bd.chan.clone()));
                    }
                }
            },
            U::N(n) => match n {
                Name::Bound(i) => {
                    let id = binder_name(&binders, i);
                    let x = b.name_var(&id);
                    names.push(x);
                }
                Name::Free(l) => {
                    let id = match patframes.last() {
                        Some(f) => f.get(l as usize).cloned().unwrap_or_else(|| format!("x{l}")),
                        None => format!("f{l}"),
                    };
                    let x = b.name_var(&id);
                    names.push(x);
                }
                Name::Unforgeable(bytes) => {
                    let id = format!("u_{:02x}{:02x}{:02x}{:02x}", bytes[0], bytes[1], bytes[2], bytes[3]);
                    let x = b.name_var(&id);
                    names.push(x);
                }
                Name::Quote(q) => {
                    jobs.push(U::MkQuote);
                    jobs.push(U::P(q));
                }
            },
            U::PushPat(ids) => patframes.push(ids),
            U::PopPat => {
                patframes.pop();
            }
            U::PushBinders(ids) => binders.extend(ids),
            U::MkQuote => {
                let p = procs.pop().unwrap_or_else(|| b.nil());
                let x = b.quote(p);
                names.push(x);
            }
            U::MkEval => {
                let n = names.pop().unwrap_or_else(|| b.name_var("?"));
                let x = b.eval(n);
                procs.push(x);
            }
            U::MkPar(n) => {
                let at = procs.len().saturating_sub(n as usize);
                let parts = procs.split_off(at);
                let x = b.par(&parts);
                procs.push(x);
            }
            U::MkSend(persistent, n) => {
                let at = procs.len().saturating_sub(n as usize);
                let args = procs.split_off(at);
                let chan = names.pop().unwrap_or_else(|| b.name_var("?"));
                let x = b.send(chan, persistent, &args);
                procs.push(x);
            }
            U::MkNew(k) => {
                let body = procs.pop().unwrap_or_else(|| b.nil());
                let at = binders.len().saturating_sub(k as usize);
                let ids = binders.split_off(at);
                let refs: Vec<&str> = ids.iter().map(|s| s.as_str()).collect();
                let x = b.new_names(&refs, body);
                procs.push(x);
            }
            U::MkBind(kind, npats) => {
                let at = names.len().saturating_sub(npats as usize);
                let pats = names.split_off(at);
                let chan = names.pop().unwrap_or_else(|| b.name_var("?"));
                binds.push((kind, pats, chan));
            }
            U::MkRecv(nbinds, k) => {
                let body = procs.pop().unwrap_or_else(|| b.nil());
                let at = binds.len().saturating_sub(nbinds as usize);
                let bs = binds.split_off(at);
                let cut = binders.len().saturating_sub(k as usize);
                binders.truncate(cut);
                let x = b.receive(&bs, body);
                procs.push(x);
            }
        }
    }
    let root = procs.pop().unwrap_or_else(|| b.nil());
    b.finish(root)
}

fn binder_name(binders: &[String], i: u32) -> String {
    let n = binders.len();
    if (i as usize) < n {
        binders[n - 1 - i as usize].clone()
    } else {
        format!("?{i}")
    }
}

/// A term as source text.
pub fn show(t: &Norm) -> String {
    let ast = unnormalise(t);
    print_proc(&ast, ast.root(), Style::Compact)
}

/// A term as indented source text.
pub fn show_pretty(t: &Norm) -> String {
    let ast = unnormalise(t);
    print_proc(&ast, ast.root(), Style::Pretty)
}
