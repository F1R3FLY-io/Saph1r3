//! The normal form, its encoding and its decoding (spec §7.3).
//!
//! The order key is the encoding itself (Req. 7.6): parallel components and the
//! binds of one receipt are ordered by bytewise lexicographic comparison of
//! their own encodings. Encodings are built once, bottom-up, in the smart
//! constructors below, and reused as the order key of the parent (Req. 7.7).

use crate::hash::{Blake2b256, Digest32, Hash32};
use std::sync::{Arc, OnceLock};

pub use k1ndl1ng_ast::BindKind;

// Tags, in the order the node's sorter assigns its Score constants.
pub const T_NIL: u8 = 0x00;
pub const T_PAR: u8 = 0x01;
pub const T_SEND: u8 = 0x02;
pub const T_RECEIVE: u8 = 0x03;
pub const T_NEW: u8 = 0x04;
pub const T_EVAL: u8 = 0x05;
pub const T_BOUND_VAR: u8 = 0x06;
pub const T_FREE_VAR: u8 = 0x07;
pub const T_WILD: u8 = 0x08;
pub const T_QUOTE: u8 = 0x10;
pub const T_BOUND_NAME: u8 = 0x11;
pub const T_FREE_NAME: u8 = 0x12;
pub const T_UNFORGEABLE: u8 = 0x13;
pub const T_BIND: u8 = 0x20;

#[derive(Clone, PartialEq, Eq)]
pub enum Name {
    Quote(Norm),
    /// de Bruijn index of a bound name.
    Bound(u32),
    /// de Bruijn level of a free name, or, inside a pattern, the local binder slot.
    Free(u32),
    Unforgeable([u8; 32]),
}

impl Name {
    pub fn encode_into(&self, out: &mut Vec<u8>) {
        match self {
            Name::Quote(p) => {
                out.push(T_QUOTE);
                out.extend_from_slice(p.encode());
            }
            Name::Bound(i) => {
                out.push(T_BOUND_NAME);
                leb(*i, out);
            }
            Name::Free(l) => {
                out.push(T_FREE_NAME);
                leb(*l, out);
            }
            Name::Unforgeable(b) => {
                out.push(T_UNFORGEABLE);
                out.extend_from_slice(b);
            }
        }
    }
    pub fn encode(&self) -> Vec<u8> {
        let mut v = Vec::new();
        self.encode_into(&mut v);
        v
    }
    /// The content hash a `CampF1R3` channel is keyed by.
    pub fn content_hash(&self) -> Hash32 {
        Blake2b256::digest(&self.encode())
    }
    pub fn is_variable(&self) -> bool {
        !matches!(self, Name::Quote(_))
    }
}

impl std::fmt::Debug for Name {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Name::Quote(p) => write!(f, "@{p:?}"),
            Name::Bound(i) => write!(f, "n{i}"),
            Name::Free(l) => write!(f, "f{l}"),
            Name::Unforgeable(b) => write!(f, "u{:02x}{:02x}", b[0], b[1]),
        }
    }
}

#[derive(Clone, PartialEq, Eq)]
pub struct NBind {
    pub kind: BindKind,
    pub chan: Name,
    pub pats: Vec<Name>,
    /// Binder slots this bind contributes, precomputed for the matcher.
    pub binders: u16,
    enc: Vec<u8>,
}

impl NBind {
    pub fn new(kind: BindKind, chan: Name, pats: Vec<Name>, binders: u16) -> NBind {
        let mut enc = Vec::new();
        enc.push(T_BIND);
        enc.push(kind.tag());
        chan.encode_into(&mut enc);
        leb(pats.len() as u32, &mut enc);
        for p in &pats {
            p.encode_into(&mut enc);
        }
        NBind {
            kind,
            chan,
            pats,
            binders,
            enc,
        }
    }
    pub fn encode(&self) -> &[u8] {
        &self.enc
    }
}

impl std::fmt::Debug for NBind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?} {} {:?}", self.pats, self.kind.arrow(), self.chan)
    }
}

#[derive(Clone, PartialEq, Eq)]
pub enum Node {
    Nil,
    /// Two or more components, ordered by encoding, no `Nil` among them.
    Par(Vec<Norm>),
    Send {
        chan: Name,
        persistent: bool,
        args: Vec<Norm>,
    },
    Receive {
        binds: Vec<NBind>,
        body: Norm,
    },
    New {
        count: u32,
        body: Norm,
    },
    /// `*x` where `x` is a variable; `*@P` has been rewritten to `P`.
    Eval(Name),
    BoundVar(u32),
    FreeVar(u32),
    Wild,
}

struct Inner {
    node: Node,
    enc: Vec<u8>,
    hash: OnceLock<Hash32>,
    size: u32,
    depth: u32,
}

impl Drop for Inner {
    /// Iterative dismantling: `Drop` must not recurse on term structure
    /// (Req. 4.5), because a gesture interface can nest quotes thousands deep.
    fn drop(&mut self) {
        let mut stack = Vec::new();
        take_children(&mut self.node, &mut stack);
        while let Some(mut n) = stack.pop() {
            if let Some(inner) = Arc::get_mut(&mut n.0) {
                take_children(&mut inner.node, &mut stack);
            }
            drop(n);
        }
    }
}

fn take_children(node: &mut Node, out: &mut Vec<Norm>) {
    let old = std::mem::replace(node, Node::Nil);
    match old {
        Node::Par(v) => out.extend(v),
        Node::Send { chan, args, .. } => {
            push_name(chan, out);
            out.extend(args);
        }
        Node::Receive { binds, body } => {
            for b in binds {
                push_name(b.chan, out);
                for p in b.pats {
                    push_name(p, out);
                }
            }
            out.push(body);
        }
        Node::New { body, .. } => out.push(body),
        Node::Eval(n) => push_name(n, out),
        _ => {}
    }
}

fn push_name(n: Name, out: &mut Vec<Norm>) {
    if let Name::Quote(q) = n {
        out.push(q);
    }
}

/// A term in normal form, with its encoding and (memoised) content hash.
/// Cloning is an `Arc` bump, so subterms are shared, not copied.
#[derive(Clone)]
pub struct Norm(Arc<Inner>);

impl PartialEq for Norm {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0) || self.0.enc == other.0.enc
    }
}
impl Eq for Norm {}
impl PartialOrd for Norm {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Norm {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0.enc.cmp(&other.0.enc)
    }
}

impl std::fmt::Debug for Norm {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", crate::show::show(self))
    }
}

impl Norm {
    fn make(node: Node, enc: Vec<u8>, size: u32, depth: u32) -> Norm {
        Norm(Arc::new(Inner {
            node,
            enc,
            hash: OnceLock::new(),
            size,
            depth,
        }))
    }

    pub fn node(&self) -> &Node {
        &self.0.node
    }
    /// The encoding. Memoised: built once, bottom-up (Req. 7.7).
    pub fn encode(&self) -> &[u8] {
        &self.0.enc
    }
    /// The content hash, computed exactly once, from the encoding (Req. 7.10).
    pub fn hash(&self) -> Hash32 {
        *self.0.hash.get_or_init(|| Blake2b256::digest(&self.0.enc))
    }
    pub fn size(&self) -> u32 {
        self.0.size
    }
    pub fn depth(&self) -> u32 {
        self.0.depth
    }
    pub fn is_nil(&self) -> bool {
        matches!(self.0.node, Node::Nil)
    }
    /// The par components: what rho reduces at.
    pub fn top_level(&self) -> Vec<Norm> {
        match &self.0.node {
            Node::Nil => Vec::new(),
            Node::Par(v) => v.clone(),
            _ => vec![self.clone()],
        }
    }

    // ---- smart constructors -------------------------------------------------

    pub fn nil() -> Norm {
        Norm::make(Node::Nil, vec![T_NIL], 1, 1)
    }

    /// Parallel composition: flattened, `Nil` components dropped, components
    /// ordered by their encodings. This is what absorbs commutativity,
    /// associativity and the unit law (Prop. 7.8).
    pub fn par(parts: Vec<Norm>) -> Norm {
        let mut flat: Vec<Norm> = Vec::with_capacity(parts.len());
        let mut work: Vec<Norm> = parts;
        work.reverse();
        while let Some(p) = work.pop() {
            match p.node() {
                Node::Nil => {}
                Node::Par(v) => {
                    // Splice, iteratively: nested pars can arise from substitution.
                    for c in v.iter().rev() {
                        work.push(c.clone());
                    }
                }
                _ => flat.push(p),
            }
        }
        match flat.len() {
            0 => Norm::nil(),
            1 => flat.pop().unwrap(),
            _ => {
                flat.sort();
                let mut enc = vec![T_PAR];
                leb(flat.len() as u32, &mut enc);
                let mut size = 1u32;
                let mut depth = 0u32;
                for p in &flat {
                    enc.extend_from_slice(p.encode());
                    size = size.saturating_add(p.size());
                    depth = depth.max(p.depth());
                }
                Norm::make(Node::Par(flat), enc, size, depth + 1)
            }
        }
    }

    pub fn send(chan: Name, persistent: bool, args: Vec<Norm>) -> Norm {
        let mut enc = vec![T_SEND];
        chan.encode_into(&mut enc);
        enc.push(persistent as u8);
        leb(args.len() as u32, &mut enc);
        let mut size = 1u32;
        let mut depth = 0u32;
        if let Name::Quote(q) = &chan {
            size = size.saturating_add(q.size());
            depth = depth.max(q.depth());
        }
        for a in &args {
            enc.extend_from_slice(a.encode());
            size = size.saturating_add(a.size());
            depth = depth.max(a.depth());
        }
        Norm::make(
            Node::Send {
                chan,
                persistent,
                args,
            },
            enc,
            size,
            depth + 1,
        )
    }

    /// The binds of one receipt are ordered by their encodings (Req. 7.6).
    /// Callers that assign binder slots must sort *before* assigning, which
    /// `sort_binds` below does for them.
    pub fn receive(binds: Vec<NBind>, body: Norm) -> Norm {
        let mut binds = binds;
        binds.sort_by(|a, b| a.encode().cmp(b.encode()));
        let mut enc = vec![T_RECEIVE];
        leb(binds.len() as u32, &mut enc);
        let mut size = 1u32;
        let mut depth = body.depth();
        for b in &binds {
            enc.extend_from_slice(b.encode());
            size = size.saturating_add(1 + b.pats.len() as u32);
            if let Name::Quote(q) = &b.chan {
                size = size.saturating_add(q.size());
                depth = depth.max(q.depth());
            }
            for p in &b.pats {
                if let Name::Quote(q) = p {
                    size = size.saturating_add(q.size());
                    depth = depth.max(q.depth());
                }
            }
        }
        enc.extend_from_slice(body.encode());
        size = size.saturating_add(body.size());
        Norm::make(Node::Receive { binds, body }, enc, size, depth + 1)
    }

    pub fn new_scope(count: u32, body: Norm) -> Norm {
        if count == 0 {
            return body;
        }
        let mut enc = vec![T_NEW];
        leb(count, &mut enc);
        enc.extend_from_slice(body.encode());
        let size = body.size().saturating_add(1);
        let depth = body.depth() + 1;
        Norm::make(Node::New { count, body }, enc, size, depth)
    }

    /// `*x`. Applies the rewrite `*@P = P` (Prop. 7.8).
    pub fn eval(name: Name) -> Norm {
        match name {
            Name::Quote(p) => p,
            other => {
                let mut enc = vec![T_EVAL];
                other.encode_into(&mut enc);
                Norm::make(Node::Eval(other), enc, 2, 2)
            }
        }
    }

    pub fn bound_var(i: u32) -> Norm {
        let mut enc = vec![T_BOUND_VAR];
        leb(i, &mut enc);
        Norm::make(Node::BoundVar(i), enc, 1, 1)
    }
    pub fn free_var(l: u32) -> Norm {
        let mut enc = vec![T_FREE_VAR];
        leb(l, &mut enc);
        Norm::make(Node::FreeVar(l), enc, 1, 1)
    }
    pub fn wild() -> Norm {
        Norm::make(Node::Wild, vec![T_WILD], 1, 1)
    }

    /// `@P`. Applies the rewrite `@*x = x` (Prop. 7.8).
    pub fn quote(p: Norm) -> Name {
        if let Node::Eval(n) = p.node() {
            return n.clone();
        }
        Name::Quote(p)
    }
}

/// Sort binds by encoding and return the permutation, so a caller assigning
/// binder slots can follow the same order the encoding will.
pub fn sort_binds(binds: &mut [NBind]) -> Vec<usize> {
    let mut ix: Vec<usize> = (0..binds.len()).collect();
    ix.sort_by(|a, b| binds[*a].encode().cmp(binds[*b].encode()));
    let mut sorted: Vec<Option<NBind>> = binds.iter().cloned().map(Some).collect();
    for (target, source) in ix.iter().enumerate() {
        binds[target] = sorted[*source].take().unwrap();
    }
    ix
}

#[inline]
pub fn leb(mut v: u32, out: &mut Vec<u8>) {
    loop {
        let byte = (v & 0x7f) as u8;
        v >>= 7;
        if v == 0 {
            out.push(byte);
            return;
        }
        out.push(byte | 0x80);
    }
}

// ---------------------------------------------------------------------------
// Decoding. Iterative; a function, as Prop. 7.8's second direction requires.

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DecodeError {
    Truncated,
    BadTag(u8),
    BadLeb,
    TooDeep,
    Trailing,
}

struct Reader<'a> {
    b: &'a [u8],
    i: usize,
}

impl<'a> Reader<'a> {
    fn byte(&mut self) -> Result<u8, DecodeError> {
        let c = *self.b.get(self.i).ok_or(DecodeError::Truncated)?;
        self.i += 1;
        Ok(c)
    }
    fn leb(&mut self) -> Result<u32, DecodeError> {
        let mut out = 0u32;
        let mut shift = 0u32;
        loop {
            let c = self.byte()?;
            if shift > 28 {
                return Err(DecodeError::BadLeb);
            }
            out |= ((c & 0x7f) as u32) << shift;
            if c & 0x80 == 0 {
                return Ok(out);
            }
            shift += 7;
        }
    }
    fn take32(&mut self) -> Result<[u8; 32], DecodeError> {
        if self.i + 32 > self.b.len() {
            return Err(DecodeError::Truncated);
        }
        let mut out = [0u8; 32];
        out.copy_from_slice(&self.b[self.i..self.i + 32]);
        self.i += 32;
        Ok(out)
    }
}

enum Step {
    Proc,
    Name,
    SendAfterChan,
    Bind,
    BindAfterChan(BindKind),
    MkPar(u32),
    MkSend(bool, u32),
    MkReceive(u32),
    MkNew(u32),
    MkEval,
    MkQuote,
    MkBind(BindKind, u32),
}

const MAX_DECODE_DEPTH: usize = 1 << 20;

impl Norm {
    /// Decode a term from its encoding. Total: every byte string either decodes
    /// or returns an error (Req. 8.2). Iterative, in stream order.
    pub fn decode(bytes: &[u8]) -> Result<Norm, DecodeError> {
        let mut r = Reader { b: bytes, i: 0 };
        let t = decode_at(&mut r)?;
        if r.i != bytes.len() {
            return Err(DecodeError::Trailing);
        }
        Ok(t)
    }
}

fn kind_of(b: u8) -> Result<BindKind, DecodeError> {
    match b {
        0 => Ok(BindKind::Linear),
        1 => Ok(BindKind::Persistent),
        2 => Ok(BindKind::Peek),
        other => Err(DecodeError::BadTag(other)),
    }
}

/// Binder slots of a pattern list: one more than the largest local free level.
pub fn binder_count(pats: &[Name]) -> u16 {
    let mut top: Option<u32> = None;
    let mut names: Vec<Name> = pats.to_vec();
    let mut procs: Vec<Norm> = Vec::new();
    loop {
        while let Some(n) = names.pop() {
            match n {
                Name::Free(l) => top = Some(top.map_or(l, |t: u32| t.max(l))),
                Name::Quote(q) => procs.push(q),
                _ => {}
            }
        }
        let p = match procs.pop() {
            Some(p) => p,
            None => break,
        };
        match p.node() {
            Node::FreeVar(l) => top = Some(top.map_or(*l, |t: u32| t.max(*l))),
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
    top.map_or(0, |t| (t + 1) as u16)
}

fn decode_at(r: &mut Reader<'_>) -> Result<Norm, DecodeError> {
    let mut steps: Vec<Step> = vec![Step::Proc];
    let mut procs: Vec<Norm> = Vec::new();
    let mut names: Vec<Name> = Vec::new();
    let mut binds: Vec<NBind> = Vec::new();
    while let Some(step) = steps.pop() {
        if steps.len() > MAX_DECODE_DEPTH {
            return Err(DecodeError::TooDeep);
        }
        match step {
            Step::Proc => {
                let tag = r.byte()?;
                match tag {
                    T_NIL => procs.push(Norm::nil()),
                    T_WILD => procs.push(Norm::wild()),
                    T_BOUND_VAR => {
                        let i = r.leb()?;
                        procs.push(Norm::bound_var(i));
                    }
                    T_FREE_VAR => {
                        let l = r.leb()?;
                        procs.push(Norm::free_var(l));
                    }
                    T_PAR => {
                        let n = r.leb()?;
                        steps.push(Step::MkPar(n));
                        for _ in 0..n {
                            steps.push(Step::Proc);
                        }
                    }
                    T_SEND => {
                        steps.push(Step::SendAfterChan);
                        steps.push(Step::Name);
                    }
                    T_RECEIVE => {
                        let m = r.leb()?;
                        steps.push(Step::MkReceive(m));
                        steps.push(Step::Proc);
                        for _ in 0..m {
                            steps.push(Step::Bind);
                        }
                    }
                    T_NEW => {
                        let k = r.leb()?;
                        steps.push(Step::MkNew(k));
                        steps.push(Step::Proc);
                    }
                    T_EVAL => {
                        steps.push(Step::MkEval);
                        steps.push(Step::Name);
                    }
                    other => return Err(DecodeError::BadTag(other)),
                }
            }
            Step::Name => {
                let tag = r.byte()?;
                match tag {
                    T_BOUND_NAME => names.push(Name::Bound(r.leb()?)),
                    T_FREE_NAME => names.push(Name::Free(r.leb()?)),
                    T_UNFORGEABLE => names.push(Name::Unforgeable(r.take32()?)),
                    T_QUOTE => {
                        steps.push(Step::MkQuote);
                        steps.push(Step::Proc);
                    }
                    other => return Err(DecodeError::BadTag(other)),
                }
            }
            Step::SendAfterChan => {
                let persistent = r.byte()? != 0;
                let n = r.leb()?;
                steps.push(Step::MkSend(persistent, n));
                for _ in 0..n {
                    steps.push(Step::Proc);
                }
            }
            Step::Bind => {
                let tag = r.byte()?;
                if tag != T_BIND {
                    return Err(DecodeError::BadTag(tag));
                }
                let kind = kind_of(r.byte()?)?;
                steps.push(Step::BindAfterChan(kind));
                steps.push(Step::Name);
            }
            Step::BindAfterChan(kind) => {
                let n = r.leb()?;
                steps.push(Step::MkBind(kind, n));
                for _ in 0..n {
                    steps.push(Step::Name);
                }
            }
            Step::MkBind(kind, n) => {
                let at = names
                    .len()
                    .checked_sub(n as usize)
                    .ok_or(DecodeError::Truncated)?;
                let pats: Vec<Name> = names.split_off(at);
                let chan = names.pop().ok_or(DecodeError::Truncated)?;
                let bc = binder_count(&pats);
                binds.push(NBind::new(kind, chan, pats, bc));
            }
            Step::MkReceive(m) => {
                let body = procs.pop().ok_or(DecodeError::Truncated)?;
                let at = binds
                    .len()
                    .checked_sub(m as usize)
                    .ok_or(DecodeError::Truncated)?;
                let bs: Vec<NBind> = binds.split_off(at);
                procs.push(Norm::receive(bs, body));
            }
            Step::MkQuote => {
                let p = procs.pop().ok_or(DecodeError::Truncated)?;
                names.push(Norm::quote(p));
            }
            Step::MkEval => {
                let n = names.pop().ok_or(DecodeError::Truncated)?;
                procs.push(Norm::eval(n));
            }
            Step::MkPar(n) => {
                let at = procs
                    .len()
                    .checked_sub(n as usize)
                    .ok_or(DecodeError::Truncated)?;
                let parts: Vec<Norm> = procs.split_off(at);
                procs.push(Norm::par(parts));
            }
            Step::MkSend(persistent, n) => {
                let at = procs
                    .len()
                    .checked_sub(n as usize)
                    .ok_or(DecodeError::Truncated)?;
                let args: Vec<Norm> = procs.split_off(at);
                let chan = names.pop().ok_or(DecodeError::Truncated)?;
                procs.push(Norm::send(chan, persistent, args));
            }
            Step::MkNew(k) => {
                let body = procs.pop().ok_or(DecodeError::Truncated)?;
                procs.push(Norm::new_scope(k, body));
            }
        }
    }
    procs.pop().ok_or(DecodeError::Truncated)
}
