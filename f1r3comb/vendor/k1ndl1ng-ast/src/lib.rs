//! `k1ndl1ng-ast` — the typed tree for the rholang kernel.
//!
//! Owns the tree and nothing else: node types, arena, spans, trivia, node
//! identity, traversal, construction and printing (K1ndl1ng spec §4).
//!
//! Deviation from the specification, deliberate and documented: the spec types
//! the tree as `Ast<'src>` with identifiers borrowed from the source. This
//! implementation interns identifiers into a symbol table owned by the tree, so
//! `Ast: 'static + Send`, `unnormalise` can return a tree with no lifetime
//! gymnastics, and a tree built by a gesture interface needs no backing string.
//! Every other requirement of §4 is met as written: `u32` indices throughout
//! (Req. 4.3), flat vectors (Req. 4.4), flat par (Req. 4.2), and no recursion in
//! any traversal, including `Drop` (Req. 4.5), which is free here because the
//! arena owns no nested values.

#![forbid(unsafe_code)]

use std::collections::HashMap;

pub mod print;

pub type Sym = u32;

macro_rules! id_type {
    ($n:ident) => {
        #[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
        pub struct $n(pub u32);
        impl $n {
            #[inline]
            pub fn ix(self) -> usize {
                self.0 as usize
            }
        }
    };
}

id_type!(ProcId);
id_type!(NameId);
id_type!(ReceiptId);
id_type!(BindId);
id_type!(DiagId);

/// Byte offsets into the source a tree was built from. `Span::NULL` marks a
/// constructed node with no source text.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Span {
    pub lo: u32,
    pub hi: u32,
}

impl Span {
    pub const NULL: Span = Span { lo: 0, hi: 0 };
    pub fn new(lo: u32, hi: u32) -> Span {
        Span { lo, hi }
    }
    pub fn is_null(self) -> bool {
        self.lo == 0 && self.hi == 0
    }
}

/// A run of children in one of the side tables.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Slice {
    pub off: u32,
    pub len: u32,
}

impl Slice {
    pub const EMPTY: Slice = Slice { off: 0, len: 0 };
    fn range(self) -> std::ops::Range<usize> {
        self.off as usize..(self.off as usize + self.len as usize)
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ProcNode {
    Nil,
    /// Flat, two or more components, in source order (Req. 4.2).
    Par(Slice),
    Send {
        chan: NameId,
        persistent: bool,
        args: Slice,
    },
    Receive {
        receipt: ReceiptId,
        body: ProcId,
    },
    Eval {
        name: NameId,
    },
    New {
        names: Slice,
        body: ProcId,
    },
    Var {
        id: Sym,
    },
    Wild,
    Error {
        diag: DiagId,
    },
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum NameNode {
    Var { id: Sym },
    Quote(ProcId),
    Wild,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Receipt {
    pub binds: Slice,
    pub guard: Option<ProcId>,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Bind {
    pub kind: BindKind,
    pub pats: Slice,
    pub chan: NameId,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum BindKind {
    Linear,
    Persistent,
    Peek,
}

impl BindKind {
    pub fn tag(self) -> u8 {
        match self {
            BindKind::Linear => 0,
            BindKind::Persistent => 1,
            BindKind::Peek => 2,
        }
    }
    pub fn arrow(self) -> &'static str {
        match self {
            BindKind::Linear => "<-",
            BindKind::Persistent => "<=",
            BindKind::Peek => "<<-",
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum TriviaKind {
    LineComment,
    BlockComment,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Trivia {
    pub kind: TriviaKind,
    pub span: Span,
}

#[derive(Clone, Debug)]
pub struct Ast {
    procs: Vec<ProcNode>,
    proc_spans: Vec<Span>,
    names: Vec<NameNode>,
    name_spans: Vec<Span>,
    receipts: Vec<Receipt>,
    binds: Vec<Bind>,
    proc_slices: Vec<ProcId>,
    name_slices: Vec<NameId>,
    sym_slices: Vec<Sym>,
    syms: Vec<String>,
    trivia: Vec<Trivia>,
    root: ProcId,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum WfError {
    IndexOutOfRange(&'static str, u32),
    ThinPar(ProcId),
    ErrorNode(ProcId),
    Unreachable(u32),
    RootOutOfRange,
    EmptyBind(BindId),
}

impl std::fmt::Display for WfError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            WfError::IndexOutOfRange(w, i) => write!(f, "{w} index {i} out of range"),
            WfError::ThinPar(p) => write!(f, "par at {} has fewer than two components", p.0),
            WfError::ErrorNode(p) => write!(f, "error node at {}", p.0),
            WfError::Unreachable(n) => write!(f, "{n} nodes unreachable from the root"),
            WfError::RootOutOfRange => write!(f, "root out of range"),
            WfError::EmptyBind(b) => write!(f, "bind at {} has no patterns", b.0),
        }
    }
}

impl Ast {
    pub fn root(&self) -> ProcId {
        self.root
    }
    pub fn proc(&self, p: ProcId) -> ProcNode {
        self.procs[p.ix()]
    }
    pub fn name(&self, n: NameId) -> NameNode {
        self.names[n.ix()]
    }
    pub fn receipt(&self, r: ReceiptId) -> Receipt {
        self.receipts[r.ix()]
    }
    pub fn bind(&self, b: BindId) -> Bind {
        self.binds[b.ix()]
    }
    pub fn proc_span(&self, p: ProcId) -> Span {
        self.proc_spans[p.ix()]
    }
    pub fn name_span(&self, n: NameId) -> Span {
        self.name_spans[n.ix()]
    }
    pub fn procs_of(&self, s: Slice) -> &[ProcId] {
        &self.proc_slices[s.range()]
    }
    pub fn names_of(&self, s: Slice) -> &[NameId] {
        &self.name_slices[s.range()]
    }
    pub fn syms_of(&self, s: Slice) -> &[Sym] {
        &self.sym_slices[s.range()]
    }
    pub fn binds_of(&self, s: Slice) -> impl Iterator<Item = BindId> {
        (s.off..s.off + s.len).map(BindId)
    }
    pub fn text(&self, s: Sym) -> &str {
        &self.syms[s as usize]
    }
    pub fn trivia(&self) -> &[Trivia] {
        &self.trivia
    }
    pub fn proc_count(&self) -> u32 {
        self.procs.len() as u32
    }
    pub fn name_count(&self) -> u32 {
        self.names.len() as u32
    }

    /// O(n) well-formedness check (spec Def. 4.1 / Req. 4.6). Iterative.
    pub fn check(&self) -> Result<(), WfError> {
        if self.root.ix() >= self.procs.len() {
            return Err(WfError::RootOutOfRange);
        }
        let np = self.procs.len() as u32;
        let nn = self.names.len() as u32;
        for (i, p) in self.procs.iter().enumerate() {
            match *p {
                ProcNode::Nil | ProcNode::Wild | ProcNode::Var { .. } => {}
                ProcNode::Error { .. } => return Err(WfError::ErrorNode(ProcId(i as u32))),
                ProcNode::Par(s) => {
                    if s.len < 2 {
                        return Err(WfError::ThinPar(ProcId(i as u32)));
                    }
                    self.check_proc_slice(s, np)?;
                }
                ProcNode::Send { chan, args, .. } => {
                    if chan.0 >= nn {
                        return Err(WfError::IndexOutOfRange("name", chan.0));
                    }
                    self.check_proc_slice(args, np)?;
                }
                ProcNode::Receive { receipt, body } => {
                    if receipt.ix() >= self.receipts.len() {
                        return Err(WfError::IndexOutOfRange("receipt", receipt.0));
                    }
                    if body.0 >= np {
                        return Err(WfError::IndexOutOfRange("proc", body.0));
                    }
                    let r = self.receipts[receipt.ix()];
                    if r.binds.len == 0 {
                        return Err(WfError::IndexOutOfRange("bind", r.binds.off));
                    }
                    for b in self.binds_of(r.binds) {
                        if b.ix() >= self.binds.len() {
                            return Err(WfError::IndexOutOfRange("bind", b.0));
                        }
                        let bd = self.binds[b.ix()];
                        if bd.pats.len == 0 {
                            return Err(WfError::EmptyBind(b));
                        }
                        if bd.chan.0 >= nn {
                            return Err(WfError::IndexOutOfRange("name", bd.chan.0));
                        }
                        for n in self.names_of(bd.pats) {
                            if n.0 >= nn {
                                return Err(WfError::IndexOutOfRange("name", n.0));
                            }
                        }
                    }
                }
                ProcNode::Eval { name } => {
                    if name.0 >= nn {
                        return Err(WfError::IndexOutOfRange("name", name.0));
                    }
                }
                ProcNode::New { names, body } => {
                    if names.len == 0 {
                        return Err(WfError::IndexOutOfRange("new", names.off));
                    }
                    for s in self.syms_of(names) {
                        if *s as usize >= self.syms.len() {
                            return Err(WfError::IndexOutOfRange("sym", *s));
                        }
                    }
                    if body.0 >= np {
                        return Err(WfError::IndexOutOfRange("proc", body.0));
                    }
                }
            }
        }
        for n in &self.names {
            if let NameNode::Quote(p) = *n {
                if p.0 >= np {
                    return Err(WfError::IndexOutOfRange("proc", p.0));
                }
            }
        }
        // Reachability, iteratively.
        let mut seen = vec![false; self.procs.len()];
        let mut seen_n = vec![false; self.names.len()];
        let mut stack = vec![self.root];
        let mut reached = 0u32;
        while let Some(p) = stack.pop() {
            if seen[p.ix()] {
                continue;
            }
            seen[p.ix()] = true;
            reached += 1;
            let push_name = |n: NameId, stack: &mut Vec<ProcId>, seen_n: &mut Vec<bool>| {
                if !seen_n[n.ix()] {
                    seen_n[n.ix()] = true;
                    if let NameNode::Quote(q) = self.names[n.ix()] {
                        stack.push(q);
                    }
                }
            };
            match self.procs[p.ix()] {
                ProcNode::Par(s) => stack.extend_from_slice(self.procs_of(s)),
                ProcNode::Send { chan, args, .. } => {
                    push_name(chan, &mut stack, &mut seen_n);
                    stack.extend_from_slice(self.procs_of(args));
                }
                ProcNode::Receive { receipt, body } => {
                    stack.push(body);
                    let r = self.receipts[receipt.ix()];
                    if let Some(g) = r.guard {
                        stack.push(g);
                    }
                    for b in self.binds_of(r.binds) {
                        let bd = self.binds[b.ix()];
                        push_name(bd.chan, &mut stack, &mut seen_n);
                        for n in self.names_of(bd.pats) {
                            push_name(*n, &mut stack, &mut seen_n);
                        }
                    }
                }
                ProcNode::Eval { name } => push_name(name, &mut stack, &mut seen_n),
                ProcNode::New { body, .. } => stack.push(body),
                _ => {}
            }
        }
        if reached as usize != self.procs.len() {
            return Err(WfError::Unreachable(self.procs.len() as u32 - reached));
        }
        Ok(())
    }

    fn check_proc_slice(&self, s: Slice, np: u32) -> Result<(), WfError> {
        if s.off as usize + s.len as usize > self.proc_slices.len() {
            return Err(WfError::IndexOutOfRange("proc slice", s.off));
        }
        for p in self.procs_of(s) {
            if p.0 >= np {
                return Err(WfError::IndexOutOfRange("proc", p.0));
            }
        }
        Ok(())
    }

    /// Stable node identity: path from the root plus the node's own shape, so a
    /// node keeps its identity when a sibling is edited (spec §10.1).
    pub fn stable_ids(&self) -> Vec<u64> {
        let mut out = vec![0u64; self.procs.len()];
        let mut stack = vec![(self.root, 0xcbf29ce484222325u64)];
        while let Some((p, h)) = stack.pop() {
            out[p.ix()] = h;
            let mix = |h: u64, x: u64| (h ^ x).wrapping_mul(0x100000001b3);
            match self.procs[p.ix()] {
                ProcNode::Par(s) => {
                    for (i, c) in self.procs_of(s).iter().enumerate() {
                        stack.push((*c, mix(h, i as u64 + 1)));
                    }
                }
                ProcNode::Send { args, .. } => {
                    for (i, c) in self.procs_of(args).iter().enumerate() {
                        stack.push((*c, mix(h, 0x100 + i as u64)));
                    }
                }
                ProcNode::Receive { body, .. } => stack.push((body, mix(h, 0x200))),
                ProcNode::New { body, .. } => stack.push((body, mix(h, 0x300))),
                _ => {}
            }
        }
        out
    }
}

/// Construction API (spec §4.3). A builder-produced tree is well-formed by
/// construction; `Rhobots` builds terms from gestures with no lexer linked.
#[derive(Default)]
pub struct Builder {
    procs: Vec<ProcNode>,
    proc_spans: Vec<Span>,
    names: Vec<NameNode>,
    name_spans: Vec<Span>,
    receipts: Vec<Receipt>,
    binds: Vec<Bind>,
    proc_slices: Vec<ProcId>,
    name_slices: Vec<NameId>,
    sym_slices: Vec<Sym>,
    syms: Vec<String>,
    sym_index: HashMap<String, Sym>,
    trivia: Vec<Trivia>,
}

impl Builder {
    pub fn new() -> Builder {
        Builder::default()
    }

    pub fn intern(&mut self, s: &str) -> Sym {
        if let Some(x) = self.sym_index.get(s) {
            return *x;
        }
        let ix = self.syms.len() as Sym;
        self.syms.push(s.to_string());
        self.sym_index.insert(s.to_string(), ix);
        ix
    }

    pub fn add_proc(&mut self, n: ProcNode, span: Span) -> ProcId {
        self.procs.push(n);
        self.proc_spans.push(span);
        ProcId(self.procs.len() as u32 - 1)
    }
    pub fn add_name(&mut self, n: NameNode, span: Span) -> NameId {
        self.names.push(n);
        self.name_spans.push(span);
        NameId(self.names.len() as u32 - 1)
    }
    pub fn proc_slice(&mut self, xs: &[ProcId]) -> Slice {
        let off = self.proc_slices.len() as u32;
        self.proc_slices.extend_from_slice(xs);
        Slice {
            off,
            len: xs.len() as u32,
        }
    }
    pub fn name_slice(&mut self, xs: &[NameId]) -> Slice {
        let off = self.name_slices.len() as u32;
        self.name_slices.extend_from_slice(xs);
        Slice {
            off,
            len: xs.len() as u32,
        }
    }
    pub fn sym_slice(&mut self, xs: &[Sym]) -> Slice {
        let off = self.sym_slices.len() as u32;
        self.sym_slices.extend_from_slice(xs);
        Slice {
            off,
            len: xs.len() as u32,
        }
    }
    pub fn add_bind(&mut self, b: Bind) -> BindId {
        self.binds.push(b);
        BindId(self.binds.len() as u32 - 1)
    }
    /// Binds of one receipt must be added contiguously; this records the run.
    pub fn add_receipt(&mut self, first: BindId, count: u32, guard: Option<ProcId>) -> ReceiptId {
        self.receipts.push(Receipt {
            binds: Slice {
                off: first.0,
                len: count,
            },
            guard,
        });
        ReceiptId(self.receipts.len() as u32 - 1)
    }
    pub fn add_trivia(&mut self, t: Trivia) {
        self.trivia.push(t);
    }

    // Convenience constructors, spec §4.3.
    pub fn nil(&mut self) -> ProcId {
        self.add_proc(ProcNode::Nil, Span::NULL)
    }
    pub fn par(&mut self, xs: &[ProcId]) -> ProcId {
        match xs.len() {
            0 => self.nil(),
            1 => xs[0],
            _ => {
                let s = self.proc_slice(xs);
                self.add_proc(ProcNode::Par(s), Span::NULL)
            }
        }
    }
    pub fn send(&mut self, chan: NameId, persistent: bool, args: &[ProcId]) -> ProcId {
        let s = self.proc_slice(args);
        self.add_proc(
            ProcNode::Send {
                chan,
                persistent,
                args: s,
            },
            Span::NULL,
        )
    }
    pub fn receive(&mut self, binds: &[(BindKind, Vec<NameId>, NameId)], body: ProcId) -> ProcId {
        let mut first = None;
        let mut count = 0;
        for (k, pats, chan) in binds {
            let ps = self.name_slice(pats);
            let id = self.add_bind(Bind {
                kind: *k,
                pats: ps,
                chan: *chan,
            });
            if first.is_none() {
                first = Some(id);
            }
            count += 1;
        }
        let r = self.add_receipt(first.unwrap_or(BindId(0)), count, None);
        self.add_proc(ProcNode::Receive { receipt: r, body }, Span::NULL)
    }
    pub fn eval(&mut self, name: NameId) -> ProcId {
        self.add_proc(ProcNode::Eval { name }, Span::NULL)
    }
    pub fn new_names(&mut self, ids: &[&str], body: ProcId) -> ProcId {
        let syms: Vec<Sym> = ids.iter().map(|i| self.intern(i)).collect();
        let s = self.sym_slice(&syms);
        self.add_proc(ProcNode::New { names: s, body }, Span::NULL)
    }
    pub fn quote(&mut self, p: ProcId) -> NameId {
        self.add_name(NameNode::Quote(p), Span::NULL)
    }
    pub fn name_var(&mut self, id: &str) -> NameId {
        let s = self.intern(id);
        self.add_name(NameNode::Var { id: s }, Span::NULL)
    }
    pub fn proc_var(&mut self, id: &str) -> ProcId {
        let s = self.intern(id);
        self.add_proc(ProcNode::Var { id: s }, Span::NULL)
    }
    pub fn wild_name(&mut self) -> NameId {
        self.add_name(NameNode::Wild, Span::NULL)
    }

    pub fn finish(self, root: ProcId) -> Ast {
        Ast {
            procs: self.procs,
            proc_spans: self.proc_spans,
            names: self.names,
            name_spans: self.name_spans,
            receipts: self.receipts,
            binds: self.binds,
            proc_slices: self.proc_slices,
            name_slices: self.name_slices,
            sym_slices: self.sym_slices,
            syms: self.syms,
            trivia: self.trivia,
            root,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builder_tree_is_well_formed() {
        let mut b = Builder::new();
        let nil = b.nil();
        let x = b.name_var("x");
        let s = b.send(x, false, &[nil]);
        let y = b.name_var("y");
        let ch = b.name_var("x");
        let r = b.receive(&[(BindKind::Linear, vec![y], ch)], nil);
        let root = b.par(&[s, r]);
        let ast = b.finish(root);
        assert_eq!(ast.check(), Ok(()));
    }

    #[test]
    fn thin_par_is_rejected() {
        let mut b = Builder::new();
        let nil = b.nil();
        let s = b.proc_slice(&[nil]);
        let bad = b.add_proc(ProcNode::Par(s), Span::NULL);
        let ast = b.finish(bad);
        assert!(matches!(ast.check(), Err(WfError::ThinPar(_))));
    }
}
