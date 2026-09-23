//! The normaliser: a well-formed tree to a normal form (spec §7).
//!
//! Binder resolution, de Bruijn assignment, ordering of parallel components,
//! and the static checks of Req. 7.2. One iterative walk with an explicit job
//! stack, an explicit scope stack, and an explicit stack of pattern frames.

use crate::term::*;
use k1ndl1ng_ast::*;
use std::collections::HashMap;

#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub enum Sort {
    Name,
    Proc,
}

#[derive(Clone, Debug)]
pub struct FreeName {
    pub level: u32,
    pub sort: Sort,
    pub ident: String,
    pub span: Span,
}

#[derive(Clone, Debug)]
pub struct Diag {
    pub span: Span,
    pub code: &'static str,
    pub message: String,
}

impl std::fmt::Display for Diag {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}..{}: {} [{}]", self.span.lo, self.span.hi, self.message, self.code)
    }
}

#[derive(Clone, Debug)]
pub struct Options {
    /// Reject a term with free names (Req. 7.2 (iv)). Off by default:
    /// free names are reported, not an error (Decision 7.3).
    pub require_closed: bool,
    /// Bound on the free-level ordering fixed point (Obligation 7.11).
    pub max_rounds: u32,
}

impl Default for Options {
    fn default() -> Self {
        Options {
            require_closed: false,
            max_rounds: 8,
        }
    }
}

/// A normalised term together with what the caller needs to know about its free
/// names: `Rhobots` synthesises its outer comprehension on `stdin` from this
/// list, and a node adapter uses it to decide whether a term is deployable.
#[derive(Clone, Debug)]
pub struct Normalised {
    pub term: Norm,
    pub free: Vec<FreeName>,
}

// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq)]
enum Ctx {
    Normal,
    Pattern,
}

struct Scope {
    slots: Vec<(Sym, Sort)>,
    from_new: bool,
}

struct PatFrame {
    scope_mark: usize,
    slots: Vec<(Sym, Sort, Span)>,
}

enum J {
    P(ProcId, Ctx),
    N(NameId, Ctx),
    MkPar(u32),
    MkSend(bool, u32),
    MkEval(Span),
    MkQuote,
    MkNew(u32),
    BindStart(BindId, Ctx),
    BindMid(BindKind, u32),
    RecvMid { nbinds: u32, body: ProcId },
    RecvFinish(u32),
}

struct Norms<'a> {
    ast: &'a Ast,
    diags: Vec<Diag>,
    scopes: Vec<Scope>,
    bases: Vec<u32>,
    depth: u32,
    pats: Vec<PatFrame>,
    free: Vec<FreeName>,
    free_ix: HashMap<(Sym, Sort), u32>,
    procs: Vec<Norm>,
    names: Vec<Name>,
    binds: Vec<NBind>,
    bind_slots: Vec<Vec<(Sym, Sort)>>,
}

impl<'a> Norms<'a> {
    fn err(&mut self, span: Span, code: &'static str, message: String) {
        if self.diags.len() < 256 {
            self.diags.push(Diag { span, code, message });
        }
    }

    fn push_scope(&mut self, slots: Vec<(Sym, Sort)>, from_new: bool) {
        self.bases.push(self.depth);
        self.depth += slots.len() as u32;
        self.scopes.push(Scope { slots, from_new });
    }

    fn pop_scope(&mut self) {
        if let Some(s) = self.scopes.pop() {
            self.depth -= s.slots.len() as u32;
            self.bases.pop();
        }
    }

    /// Resolve an occurrence to a de Bruijn index, searching scopes from the
    /// innermost outward, stopping at `floor`.
    fn lookup(&self, sym: Sym, sort: Sort, floor: usize) -> Option<u32> {
        for si in (floor..self.scopes.len()).rev() {
            let base = self.bases[si];
            for (pi, (s, so)) in self.scopes[si].slots.iter().enumerate() {
                if *s == sym && *so == sort {
                    return Some(self.depth - 1 - (base + pi as u32));
                }
            }
        }
        None
    }

    fn bound_by_new(&self, sym: Sym) -> bool {
        self.scopes
            .iter()
            .any(|s| s.from_new && s.slots.iter().any(|(x, so)| *x == sym && *so == Sort::Name))
    }

    /// An identifier occurrence: bound, a pattern binder site, or free.
    fn resolve(&mut self, sym: Sym, sort: Sort, span: Span, ctx: Ctx) -> Resolved {
        if ctx == Ctx::Pattern {
            let mark = self.pats.last().map(|p| p.scope_mark).unwrap_or(0);
            if let Some(i) = self.lookup(sym, sort, mark) {
                return Resolved::Bound(i);
            }
            if self.bound_by_new(sym) {
                let name = self.ast.text(sym).to_string();
                self.err(
                    span,
                    "new-as-pattern",
                    format!("`{name}` is bound by `new` and cannot be a pattern variable"),
                );
            }
            let frame = match self.pats.last_mut() {
                Some(f) => f,
                None => return Resolved::Free(0),
            };
            if let Some(ix) = frame
                .slots
                .iter()
                .position(|(s, so, _)| *s == sym && *so == sort)
            {
                let name = self.ast.text(sym).to_string();
                self.err(
                    span,
                    "dup-pattern-var",
                    format!("`{name}` occurs more than once in one pattern"),
                );
                return Resolved::Slot(ix as u32);
            }
            let ix = frame.slots.len() as u32;
            frame.slots.push((sym, sort, span));
            Resolved::Slot(ix)
        } else {
            if let Some(i) = self.lookup(sym, sort, 0) {
                return Resolved::Bound(i);
            }
            if let Some(l) = self.free_ix.get(&(sym, sort)) {
                return Resolved::Free(*l);
            }
            let l = self.free.len() as u32;
            self.free_ix.insert((sym, sort), l);
            self.free.push(FreeName {
                level: l,
                sort,
                ident: self.ast.text(sym).to_string(),
                span,
            });
            Resolved::Free(l)
        }
    }
}

enum Resolved {
    Bound(u32),
    Free(u32),
    Slot(u32),
}

/// Normalise a well-formed tree.
pub fn normalise(tree: &Ast, opts: &Options) -> Result<Normalised, Vec<Diag>> {
    if let Err(e) = tree.check() {
        return Err(vec![Diag {
            span: Span::NULL,
            code: "wf",
            message: format!("ill-formed tree: {e}"),
        }]);
    }
    let mut st = Norms {
        ast: tree,
        diags: Vec::new(),
        scopes: Vec::new(),
        bases: Vec::new(),
        depth: 0,
        pats: Vec::new(),
        free: Vec::new(),
        free_ix: HashMap::new(),
        procs: Vec::new(),
        names: Vec::new(),
        binds: Vec::new(),
        bind_slots: Vec::new(),
    };
    let mut jobs: Vec<J> = vec![J::P(tree.root(), Ctx::Normal)];

    while let Some(job) = jobs.pop() {
        match job {
            J::P(p, ctx) => {
                let span = tree.proc_span(p);
                match tree.proc(p) {
                    ProcNode::Nil => st.procs.push(Norm::nil()),
                    ProcNode::Error { .. } => {
                        st.err(span, "error-node", "term contains a parse error".to_string());
                        st.procs.push(Norm::nil());
                    }
                    ProcNode::Wild => {
                        if ctx == Ctx::Normal {
                            st.err(
                                span,
                                "wild-outside-pattern",
                                "`_` may only appear in a pattern".to_string(),
                            );
                        }
                        st.procs.push(Norm::wild());
                    }
                    ProcNode::Var { id } => {
                        let r = st.resolve(id, Sort::Proc, span, ctx);
                        st.procs.push(match r {
                            Resolved::Bound(i) => Norm::bound_var(i),
                            Resolved::Free(l) | Resolved::Slot(l) => Norm::free_var(l),
                        });
                    }
                    ProcNode::Par(s) => {
                        let kids = tree.procs_of(s);
                        jobs.push(J::MkPar(kids.len() as u32));
                        for k in kids.iter().rev() {
                            jobs.push(J::P(*k, ctx));
                        }
                    }
                    ProcNode::Send {
                        chan,
                        persistent,
                        args,
                    } => {
                        let kids = tree.procs_of(args);
                        jobs.push(J::MkSend(persistent, kids.len() as u32));
                        for k in kids.iter().rev() {
                            jobs.push(J::P(*k, ctx));
                        }
                        jobs.push(J::N(chan, ctx));
                    }
                    ProcNode::Eval { name } => {
                        jobs.push(J::MkEval(span));
                        jobs.push(J::N(name, ctx));
                    }
                    ProcNode::New { names, body } => {
                        let ids: Vec<(Sym, Sort)> =
                            tree.syms_of(names).iter().map(|s| (*s, Sort::Name)).collect();
                        let k = ids.len() as u32;
                        st.push_scope(ids, true);
                        jobs.push(J::MkNew(k));
                        jobs.push(J::P(body, ctx));
                    }
                    ProcNode::Receive { receipt, body } => {
                        let r = tree.receipt(receipt);
                        if let Some(g) = r.guard {
                            st.err(
                                tree.proc_span(g),
                                "guard",
                                "`where` guards are K2 and not supported".to_string(),
                            );
                        }
                        let bs: Vec<BindId> = tree.binds_of(r.binds).collect();
                        jobs.push(J::RecvMid {
                            nbinds: bs.len() as u32,
                            body,
                        });
                        for b in bs.iter().rev() {
                            jobs.push(J::BindStart(*b, ctx));
                        }
                    }
                }
            }
            J::N(n, ctx) => {
                let span = tree.name_span(n);
                match tree.name(n) {
                    NameNode::Var { id } => {
                        let r = st.resolve(id, Sort::Name, span, ctx);
                        st.names.push(match r {
                            Resolved::Bound(i) => Name::Bound(i),
                            Resolved::Free(l) | Resolved::Slot(l) => Name::Free(l),
                        });
                    }
                    NameNode::Wild => {
                        if ctx == Ctx::Normal {
                            st.err(
                                span,
                                "wild-outside-pattern",
                                "`_` may only appear in a pattern".to_string(),
                            );
                        }
                        st.names.push(Name::Quote(Norm::wild()));
                    }
                    NameNode::Quote(p) => {
                        jobs.push(J::MkQuote);
                        jobs.push(J::P(p, ctx));
                    }
                }
            }
            J::MkPar(n) => {
                let at = st.procs.len() - n as usize;
                let parts = st.procs.split_off(at);
                st.procs.push(Norm::par(parts));
            }
            J::MkSend(persistent, n) => {
                let at = st.procs.len() - n as usize;
                let args = st.procs.split_off(at);
                let chan = st.names.pop().expect("send channel");
                st.procs.push(Norm::send(chan, persistent, args));
            }
            J::MkQuote => {
                let p = st.procs.pop().expect("quote operand");
                st.names.push(Norm::quote(p));
            }
            J::MkEval(span) => {
                let n = st.names.pop().expect("eval operand");
                if matches!(&n, Name::Quote(q) if matches!(q.node(), Node::Wild)) {
                    st.err(
                        span,
                        "eval-wild",
                        "`*_` has no meaning: a wildcard cannot be unquoted".to_string(),
                    );
                }
                st.procs.push(Norm::eval(n));
            }
            J::MkNew(k) => {
                let body = st.procs.pop().expect("new body");
                st.pop_scope();
                st.procs.push(Norm::new_scope(k, body));
            }
            J::BindStart(b, ctx) => {
                let bd = tree.bind(b);
                let pats: Vec<NameId> = tree.names_of(bd.pats).to_vec();
                st.pats.push(PatFrame {
                    scope_mark: st.scopes.len(),
                    slots: Vec::new(),
                });
                jobs.push(J::BindMid(bd.kind, pats.len() as u32));
                for p in pats.iter().rev() {
                    jobs.push(J::N(*p, Ctx::Pattern));
                }
                jobs.push(J::N(bd.chan, ctx));
            }
            J::BindMid(kind, npats) => {
                let at = st.names.len() - npats as usize;
                let pats = st.names.split_off(at);
                let chan = st.names.pop().expect("bind channel");
                let frame = st.pats.pop().expect("pattern frame");
                let slots: Vec<(Sym, Sort)> =
                    frame.slots.iter().map(|(s, so, _)| (*s, *so)).collect();
                st.binds
                    .push(NBind::new(kind, chan, pats, slots.len() as u16));
                st.bind_slots.push(slots);
            }
            J::RecvMid { nbinds, body } => {
                let at = st.binds.len() - nbinds as usize;
                let mut binds = st.binds.split_off(at);
                let slot_at = st.bind_slots.len() - nbinds as usize;
                let slot_lists = st.bind_slots.split_off(slot_at);
                // Order the binds first, then assign slots in that order, so the
                // body's indices agree with the encoding (Req. 7.6).
                let perm = sort_binds(&mut binds);
                let mut slots: Vec<(Sym, Sort)> = Vec::new();
                for old in &perm {
                    slots.extend(slot_lists[*old].iter().cloned());
                }
                let mut seen: HashMap<(Sym, Sort), ()> = HashMap::new();
                for s in &slots {
                    if seen.insert(*s, ()).is_some() {
                        let name = tree.text(s.0).to_string();
                        st.err(
                            Span::NULL,
                            "dup-receipt-var",
                            format!("`{name}` is bound twice in one receipt"),
                        );
                    }
                }
                let k = slots.len() as u32;
                st.push_scope(slots, false);
                st.binds.extend(binds);
                jobs.push(J::RecvFinish(nbinds));
                jobs.push(J::P(body, Ctx::Normal));
                let _ = k;
            }
            J::RecvFinish(nbinds) => {
                let body = st.procs.pop().expect("receive body");
                st.pop_scope();
                let at = st.binds.len() - nbinds as usize;
                let binds = st.binds.split_off(at);
                st.procs.push(Norm::receive(binds, body));
            }
        }
    }

    if !st.diags.is_empty() {
        return Err(st.diags);
    }
    let mut term = st.procs.pop().expect("normalised term");
    let mut free = st.free;

    // Free levels are assigned by first occurrence in a left-to-right traversal
    // of the *ordered* term, and ordering reorders the traversal. Iterate to a
    // fixed point, with a bound (Obligation 7.11).
    let mut rounds = 0;
    loop {
        let order = free_levels_in_order(&term);
        if order.iter().enumerate().all(|(i, l)| i as u32 == *l) {
            break;
        }
        rounds += 1;
        if rounds > opts.max_rounds {
            return Err(vec![Diag {
                span: Span::NULL,
                code: "levels",
                message: format!(
                    "free-level assignment did not reach a fixed point in {} rounds",
                    opts.max_rounds
                ),
            }]);
        }
        let n = free.len().max(order.len());
        let mut map: Vec<u32> = (0..n as u32).collect();
        for (new, old) in order.iter().enumerate() {
            map[*old as usize] = new as u32;
        }
        term = term.remap_free(&map);
        let mut permuted = free.clone();
        for f in free.iter() {
            let nl = map[f.level as usize];
            permuted[nl as usize] = FreeName {
                level: nl,
                sort: f.sort,
                ident: f.ident.clone(),
                span: f.span,
            };
        }
        free = permuted;
    }

    if opts.require_closed && !free.is_empty() {
        return Err(free
            .iter()
            .map(|f| Diag {
                span: f.span,
                code: "free",
                message: format!("`{}` is free, and a closed term was required", f.ident),
            })
            .collect());
    }
    Ok(Normalised { term, free })
}

/// The free levels of a term, in the order the encoding meets them. Patterns
/// are skipped: their `Free` levels are the bind's local binder slots, not
/// program-level free names.
pub fn free_levels_in_order(t: &Norm) -> Vec<u32> {
    let mut out: Vec<u32> = Vec::new();
    let mut seen: Vec<bool> = Vec::new();
    let mark = |l: u32, out: &mut Vec<u32>, seen: &mut Vec<bool>| {
        if seen.len() <= l as usize {
            seen.resize(l as usize + 1, false);
        }
        if !seen[l as usize] {
            seen[l as usize] = true;
            out.push(l);
        }
    };
    enum W {
        P(Norm),
        N(Name),
    }
    let mut stack = vec![W::P(t.clone())];
    while let Some(w) = stack.pop() {
        match w {
            W::N(n) => match n {
                Name::Free(l) => mark(l, &mut out, &mut seen),
                Name::Quote(q) => stack.push(W::P(q)),
                _ => {}
            },
            W::P(p) => match p.node() {
                Node::FreeVar(l) => mark(*l, &mut out, &mut seen),
                Node::Par(v) => {
                    for c in v.iter().rev() {
                        stack.push(W::P(c.clone()));
                    }
                }
                Node::Send { chan, args, .. } => {
                    for a in args.iter().rev() {
                        stack.push(W::P(a.clone()));
                    }
                    stack.push(W::N(chan.clone()));
                }
                Node::Receive { binds, body } => {
                    stack.push(W::P(body.clone()));
                    for b in binds.iter().rev() {
                        stack.push(W::N(b.chan.clone()));
                    }
                }
                Node::New { body, .. } => stack.push(W::P(body.clone())),
                Node::Eval(n) => stack.push(W::N(n.clone())),
                _ => {}
            },
        }
    }
    out
}
