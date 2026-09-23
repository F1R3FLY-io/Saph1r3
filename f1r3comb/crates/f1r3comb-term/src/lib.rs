//! `f1r3comb-term` — the target of the F1R3Comb compiler: the rho
//! combinators of draft 3 §3 in presentation A, with the constructor family
//! of Def. 3.8 and the context-instantiation combinator `inst` of §6.7
//! (F1R3Comb v0.6 §4).
//!
//! * Two sorts, two Rust types: [`CName`] (sort N) and [`Term`] (sort P). A
//!   `Term` is always a drop-free normal form: a flat multiset of atoms
//!   ordered by the bytewise order of their encodings (Req. 4.3, 4.10).
//! * The drop and the guard are distinct: `e(a)` is an atom; `*a` is not
//!   representable (Req. 4.1).
//! * The tag space is a shape byte plus a constructor offset (Req. 4.11):
//!   atoms `0x32..=0x3A`, `cons_A` at `0x50 + (tag(A) - 0x30)`, `cons_|` at
//!   `0x51`, `inst` `0x60`, context node `0x61`, hole `0x62`.
//! * A context is a term with holes in name positions; it occurs only as the
//!   third argument of `inst` (or of the curried `cstar`), and a hole occurs
//!   only inside a context (Req. 5.24). Holes carry a de Bruijn index
//!   counting enclosing context nodes, which is what makes nested units
//!   sound (see [`fill`]).
//! * No variables and no binder (Req. 4.13): the binder lives in
//!   `f1r3comb-ir`.
//!
//! The rule table, generated from the shape table, is in [`rules`]; the
//! address arithmetic of phase two (Def. 6.6) and the static channels are in
//! [`addr`].

#![forbid(unsafe_code)]

pub mod addr;
pub mod artefact;
pub mod lattice;
pub mod print;
pub mod rules;

use std::cmp::Ordering;
use std::collections::HashMap;
use std::fmt;
use std::sync::{Arc, OnceLock};

pub use k1ndl1ng_norm::Hash32;

/// BLAKE2b-256, the workspace's content hash.
pub fn blake2b(bytes: &[u8]) -> Hash32 {
    k1ndl1ng_norm::hash::blake2b_256(bytes)
}

/// The target tag block (F1R3Comb v0.6 Req. 4.11), disjoint from the
/// K1ndl1ng source tags `0x00..=0x20`.
pub mod tags {
    pub const NIL: u8 = 0x30;
    pub const PAR: u8 = 0x31;
    /// First atom tag; atoms `m,d,k,fw,bl,br,s,e,q` are `0x32..=0x3A`.
    pub const ATOM0: u8 = 0x32;
    /// `0x3B` is reserved in the atom block.
    pub const QUOTE: u8 = 0x40;
    /// The drop; presentation B only, rejected by the decoder under A.
    pub const DROP: u8 = 0x41;
    /// Constructor block: `cons_A` is `0x50 + (tag(A) - 0x30)`.
    pub const CONS0: u8 = 0x50;
    pub const CONS_PAR: u8 = 0x51;
    pub const INST: u8 = 0x60;
    pub const CONTEXT: u8 = 0x61;
    pub const HOLE: u8 = 0x62;
    /// The curried constructor has no tag in v0.6 (Mat v0.5 §13 item 7);
    /// this one is provisional.
    pub const CSTAR: u8 = 0x63;
}

/// Every atom shape: the nine base atoms, `cons_|`, one constructor per
/// base shape, and the two erecting shapes. The discriminant is the shape's
/// index in every shape-indexed table.
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
#[repr(u8)]
pub enum Shape {
    M,
    D,
    K,
    Fw,
    Bl,
    Br,
    S,
    E,
    Q,
    ConsPar,
    ConsM,
    ConsD,
    ConsK,
    ConsFw,
    ConsBl,
    ConsBr,
    ConsS,
    ConsE,
    ConsQ,
    Inst,
    Cstar,
}

pub const BASE: usize = 9;
pub const SHAPES: usize = 21;

/// What an argument position holds.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum Arg {
    Name,
    /// The stored process of `q`.
    Store,
    /// The context of `inst` / the template of `cstar`.
    Context,
}

impl Shape {
    pub const ALL: [Shape; SHAPES] = [
        Shape::M,
        Shape::D,
        Shape::K,
        Shape::Fw,
        Shape::Bl,
        Shape::Br,
        Shape::S,
        Shape::E,
        Shape::Q,
        Shape::ConsPar,
        Shape::ConsM,
        Shape::ConsD,
        Shape::ConsK,
        Shape::ConsFw,
        Shape::ConsBl,
        Shape::ConsBr,
        Shape::ConsS,
        Shape::ConsE,
        Shape::ConsQ,
        Shape::Inst,
        Shape::Cstar,
    ];

    /// The nine base atoms of draft 3 Def. 3.1.
    pub const BASES: [Shape; BASE] =
        [Shape::M, Shape::D, Shape::K, Shape::Fw, Shape::Bl, Shape::Br, Shape::S, Shape::E, Shape::Q];

    pub fn ix(self) -> usize {
        self as usize
    }

    pub fn from_ix(i: usize) -> Option<Shape> {
        Shape::ALL.get(i).copied()
    }

    pub fn is_base(self) -> bool {
        self.ix() < BASE
    }

    /// `cons_A` for a base shape `A` (Def. 3.8).
    pub fn cons_of(self) -> Option<Shape> {
        if self.is_base() {
            Shape::from_ix(Shape::ConsM.ix() + self.ix())
        } else {
            None
        }
    }

    /// The base shape a family member builds.
    pub fn builds(self) -> Option<Shape> {
        let i = self.ix();
        if (Shape::ConsM.ix()..=Shape::ConsQ.ix()).contains(&i) {
            Shape::from_ix(i - Shape::ConsM.ix())
        } else {
            None
        }
    }

    pub fn is_constructor(self) -> bool {
        self == Shape::ConsPar || self.builds().is_some()
    }

    /// Number of arguments, of every kind.
    pub fn arity(self) -> usize {
        match self {
            Shape::K | Shape::E => 1,
            Shape::M | Shape::Q | Shape::Fw | Shape::Bl | Shape::Br => 2,
            Shape::D | Shape::S => 3,
            Shape::ConsPar | Shape::Inst | Shape::Cstar => 3,
            s => s.builds().unwrap().arity() + 1,
        }
    }

    pub fn arg(self, j: usize) -> Arg {
        match (self, j) {
            (Shape::Q, 1) => Arg::Store,
            (Shape::Inst, 2) | (Shape::Cstar, 2) => Arg::Context,
            _ => Arg::Name,
        }
    }

    /// Number of name arguments.
    pub fn name_arity(self) -> usize {
        (0..self.arity()).filter(|j| self.arg(*j) == Arg::Name).count()
    }

    pub fn tag(self) -> u8 {
        match self {
            Shape::ConsPar => tags::CONS_PAR,
            Shape::Inst => tags::INST,
            Shape::Cstar => tags::CSTAR,
            s if s.is_base() => tags::ATOM0 + s.ix() as u8,
            s => tags::CONS0 + (s.builds().unwrap().tag() - tags::NIL),
        }
    }

    pub fn from_tag(t: u8) -> Option<Shape> {
        Shape::ALL.iter().copied().find(|s| s.tag() == t)
    }

    pub fn name(self) -> &'static str {
        match self {
            Shape::M => "m",
            Shape::D => "d",
            Shape::K => "k",
            Shape::Fw => "fw",
            Shape::Bl => "bl",
            Shape::Br => "br",
            Shape::S => "s",
            Shape::E => "e",
            Shape::Q => "q",
            Shape::ConsPar => "cons",
            Shape::ConsM => "cons_m",
            Shape::ConsD => "cons_d",
            Shape::ConsK => "cons_k",
            Shape::ConsFw => "cons_fw",
            Shape::ConsBl => "cons_bl",
            Shape::ConsBr => "cons_br",
            Shape::ConsS => "cons_s",
            Shape::ConsE => "cons_e",
            Shape::ConsQ => "cons_q",
            Shape::Inst => "inst",
            Shape::Cstar => "cstar",
        }
    }

    pub fn from_name(s: &str) -> Option<Shape> {
        Shape::ALL.iter().copied().find(|x| x.name() == s)
    }
}

// ---------------------------------------------------------------------------
// LEB128

pub fn leb(mut v: u64, out: &mut Vec<u8>) {
    loop {
        let b = (v & 0x7f) as u8;
        v >>= 7;
        if v == 0 {
            out.push(b);
            return;
        }
        out.push(b | 0x80);
    }
}

pub fn read_leb(bytes: &[u8], pos: &mut usize) -> Option<u64> {
    let mut v: u64 = 0;
    let mut shift = 0u32;
    loop {
        let b = *bytes.get(*pos)?;
        *pos += 1;
        if shift >= 64 {
            return None;
        }
        v |= ((b & 0x7f) as u64) << shift;
        if b & 0x80 == 0 {
            return Some(v);
        }
        shift += 7;
    }
}

// ---------------------------------------------------------------------------
// Names

/// A target name, sort N: the quotation of a drop-free normal form, or — only
/// inside a context — a hole.
#[derive(Clone)]
pub struct CName(Arc<NameInner>);

enum NameBody {
    Quote(Term),
    /// De Bruijn index: 0 is bound by the innermost enclosing context node.
    Hole(u32),
}

struct NameInner {
    body: NameBody,
    enc: Box<[u8]>,
    /// Context levels this name needs from outside (0 = closed).
    free: u32,
    hash: OnceLock<Hash32>,
}

impl CName {
    /// `@p`. Since every `Term` is drop-free, `@*a = a` holds on the nose.
    pub fn quote(p: Term) -> CName {
        let mut enc = Vec::with_capacity(1 + p.encode().len());
        enc.push(tags::QUOTE);
        enc.extend_from_slice(p.encode());
        let free = p.free_holes();
        CName(Arc::new(NameInner { body: NameBody::Quote(p), enc: enc.into_boxed_slice(), free, hash: OnceLock::new() }))
    }

    /// A hole: legal only beneath a context node.
    pub fn hole(k: u32) -> CName {
        let mut enc = vec![tags::HOLE];
        leb(k as u64, &mut enc);
        CName(Arc::new(NameInner { body: NameBody::Hole(k), enc: enc.into_boxed_slice(), free: k + 1, hash: OnceLock::new() }))
    }

    /// `@0`.
    pub fn nil() -> CName {
        static NIL: OnceLock<CName> = OnceLock::new();
        NIL.get_or_init(|| CName::quote(Term::nil())).clone()
    }

    pub fn is_hole(&self) -> Option<u32> {
        match self.0.body {
            NameBody::Hole(k) => Some(k),
            _ => None,
        }
    }

    /// The normal form of `*a`: the process the name quotes (Req. 4.3). A
    /// hole quotes nothing and drops to `Nil`; holes never reach a machine.
    pub fn drop(&self) -> Term {
        match &self.0.body {
            NameBody::Quote(t) => t.clone(),
            NameBody::Hole(_) => Term::nil(),
        }
    }

    pub fn proc_ref(&self) -> Option<&Term> {
        match &self.0.body {
            NameBody::Quote(t) => Some(t),
            NameBody::Hole(_) => None,
        }
    }

    pub fn encode(&self) -> &[u8] {
        &self.0.enc
    }

    /// Context levels needed from outside; 0 for a closed name.
    pub fn free_holes(&self) -> u32 {
        self.0.free
    }

    pub fn is_closed(&self) -> bool {
        self.0.free == 0
    }

    /// BLAKE2b-256 of the encoding; the channel identity (Req. 4.12).
    pub fn content_hash(&self) -> Hash32 {
        *self.0.hash.get_or_init(|| blake2b(&self.0.enc))
    }

    pub fn ptr_eq(&self, o: &CName) -> bool {
        Arc::ptr_eq(&self.0, &o.0)
    }

    /// Size of the encoding: the measure of Lem. 6.14's freshness by size.
    pub fn size(&self) -> usize {
        self.0.enc.len()
    }
}

impl PartialEq for CName {
    fn eq(&self, o: &Self) -> bool {
        self.ptr_eq(o) || self.encode() == o.encode()
    }
}
impl Eq for CName {}
impl PartialOrd for CName {
    fn partial_cmp(&self, o: &Self) -> Option<Ordering> {
        Some(self.cmp(o))
    }
}
impl Ord for CName {
    fn cmp(&self, o: &Self) -> Ordering {
        self.encode().cmp(o.encode())
    }
}
impl std::hash::Hash for CName {
    fn hash<H: std::hash::Hasher>(&self, h: &mut H) {
        self.encode().hash(h)
    }
}
impl fmt::Debug for CName {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", print::name_label(self))
    }
}

// ---------------------------------------------------------------------------
// Atoms

/// One combinator occurrence. `names` holds the name arguments in order;
/// `aux` holds the store of `q`, or the context of `inst` / `cstar`.
#[derive(Clone)]
pub struct Atom {
    shape: Shape,
    names: Box<[CName]>,
    aux: Option<Term>,
    enc: Arc<[u8]>,
    free: u32,
}

impl Atom {
    fn build(shape: Shape, names: Vec<CName>, aux: Option<Term>) -> Atom {
        let mut enc = vec![shape.tag()];
        let mut free = 0;
        for n in &names {
            enc.extend_from_slice(n.encode());
            free = free.max(n.free_holes());
        }
        if let Some(p) = &aux {
            if shape == Shape::Q {
                enc.extend_from_slice(p.encode());
                free = free.max(p.free_holes());
            } else {
                enc.push(tags::CONTEXT);
                enc.extend_from_slice(p.encode());
                free = free.max(p.free_holes().saturating_sub(1));
            }
        }
        Atom { shape, names: names.into_boxed_slice(), aux, enc: enc.into(), free }
    }

    /// Any shape whose arguments are all names. `None` on an arity mismatch
    /// or for `q`, `inst`, `cstar`.
    pub fn new(shape: Shape, names: Vec<CName>) -> Option<Atom> {
        if shape.name_arity() != shape.arity() || names.len() != shape.arity() {
            return None;
        }
        Some(Atom::build(shape, names, None))
    }

    pub fn m(a: CName, v: CName) -> Atom {
        Atom::build(Shape::M, vec![a, v], None)
    }
    pub fn d(a: CName, b: CName, c: CName) -> Atom {
        Atom::build(Shape::D, vec![a, b, c], None)
    }
    pub fn k(a: CName) -> Atom {
        Atom::build(Shape::K, vec![a], None)
    }
    pub fn fw(a: CName, b: CName) -> Atom {
        Atom::build(Shape::Fw, vec![a, b], None)
    }
    pub fn bl(a: CName, b: CName) -> Atom {
        Atom::build(Shape::Bl, vec![a, b], None)
    }
    pub fn br(a: CName, b: CName) -> Atom {
        Atom::build(Shape::Br, vec![a, b], None)
    }
    pub fn s(a: CName, b: CName, c: CName) -> Atom {
        Atom::build(Shape::S, vec![a, b, c], None)
    }
    pub fn e(a: CName) -> Atom {
        Atom::build(Shape::E, vec![a], None)
    }
    pub fn q(a: CName, p: Term) -> Atom {
        Atom::build(Shape::Q, vec![a], Some(p))
    }
    /// `cons_|(a, b, c)`.
    pub fn cons(a: CName, b: CName, c: CName) -> Atom {
        Atom::build(Shape::ConsPar, vec![a, b, c], None)
    }
    pub fn cons_m(a: CName, b: CName, c: CName) -> Atom {
        Atom::build(Shape::ConsM, vec![a, b, c], None)
    }
    /// `inst(a, f, C)`: `inst(a,f,C) | m(a,v) -> m(f, @C[v])`.
    pub fn inst(a: CName, f: CName, ctx: Term) -> Atom {
        Atom::build(Shape::Inst, vec![a, f], Some(ctx))
    }
    /// The curried constructor `cstar(t, f, T)`; `T` must be a single atom
    /// whose arguments are paths beneath the hole or closed names.
    pub fn cstar(t: CName, f: CName, template: Term) -> Atom {
        Atom::build(Shape::Cstar, vec![t, f], Some(template))
    }
    /// A family member `cons_A(a_1..a_n, f)`.
    pub fn cons_member(built: Shape, names: Vec<CName>) -> Option<Atom> {
        Atom::new(built.cons_of()?, names)
    }

    pub fn shape(&self) -> Shape {
        self.shape
    }
    pub fn names(&self) -> &[CName] {
        &self.names
    }
    /// The subject: the first argument, for every shape.
    pub fn subject(&self) -> &CName {
        &self.names[0]
    }
    /// The stored process of `q`.
    pub fn store(&self) -> Option<&Term> {
        if self.shape == Shape::Q {
            self.aux.as_ref()
        } else {
            None
        }
    }
    /// The context of `inst` / template of `cstar`.
    pub fn context(&self) -> Option<&Term> {
        if matches!(self.shape, Shape::Inst | Shape::Cstar) {
            self.aux.as_ref()
        } else {
            None
        }
    }
    pub fn encode(&self) -> &[u8] {
        &self.enc
    }
    pub fn free_holes(&self) -> u32 {
        self.free
    }
}

impl PartialEq for Atom {
    fn eq(&self, o: &Self) -> bool {
        self.enc == o.enc
    }
}
impl Eq for Atom {}
impl PartialOrd for Atom {
    fn partial_cmp(&self, o: &Self) -> Option<Ordering> {
        Some(self.cmp(o))
    }
}
impl Ord for Atom {
    fn cmp(&self, o: &Self) -> Ordering {
        self.enc[..].cmp(&o.enc[..])
    }
}
impl fmt::Debug for Atom {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", print::atom_inline(self))
    }
}

// ---------------------------------------------------------------------------
// Terms

/// A target process, sort P: a drop-free normal form, i.e. a finite multiset
/// of atoms, kept sorted by encoding.
#[derive(Clone)]
pub struct Term(Arc<TermInner>);

struct TermInner {
    atoms: Box<[Atom]>,
    enc: Box<[u8]>,
    free: u32,
    hash: OnceLock<Hash32>,
}

impl Term {
    pub fn nil() -> Term {
        Term::from_sorted(Vec::new())
    }

    pub fn atom(a: Atom) -> Term {
        Term::from_sorted(vec![a])
    }

    /// The normal form of a multiset of atoms.
    pub fn from_atoms(mut atoms: Vec<Atom>) -> Term {
        atoms.sort();
        Term::from_sorted(atoms)
    }

    /// Parallel composition, flattened (the monoid laws).
    pub fn par(parts: Vec<Term>) -> Term {
        let mut v = Vec::new();
        for p in parts {
            v.extend(p.atoms().iter().cloned());
        }
        Term::from_atoms(v)
    }

    fn from_sorted(atoms: Vec<Atom>) -> Term {
        let enc = match atoms.len() {
            0 => vec![tags::NIL],
            1 => atoms[0].encode().to_vec(),
            n => {
                let mut e = vec![tags::PAR];
                leb(n as u64, &mut e);
                for a in &atoms {
                    e.extend_from_slice(a.encode());
                }
                e
            }
        };
        let free = atoms.iter().map(|a| a.free_holes()).max().unwrap_or(0);
        Term(Arc::new(TermInner { atoms: atoms.into_boxed_slice(), enc: enc.into_boxed_slice(), free, hash: OnceLock::new() }))
    }

    /// The components, in canonical order.
    pub fn atoms(&self) -> &[Atom] {
        &self.0.atoms
    }

    pub fn is_nil(&self) -> bool {
        self.0.atoms.is_empty()
    }

    pub fn encode(&self) -> &[u8] {
        &self.0.enc
    }

    pub fn content_hash(&self) -> Hash32 {
        *self.0.hash.get_or_init(|| blake2b(&self.0.enc))
    }

    pub fn len(&self) -> usize {
        self.0.atoms.len()
    }

    pub fn is_empty(&self) -> bool {
        self.0.atoms.is_empty()
    }

    pub fn free_holes(&self) -> u32 {
        self.0.free
    }

    /// Closed: no hole outside a context. Only closed terms are deployable.
    pub fn is_closed(&self) -> bool {
        self.0.free == 0
    }

    /// Decode a canonical encoding. Rejects source tags, the reserved drop
    /// tag, holes outside contexts, and any byte string that is not the
    /// canonical encoding of its own decoding.
    pub fn decode(bytes: &[u8]) -> Result<Term, DecodeError> {
        let mut pos = 0usize;
        let t = decode_term(bytes, &mut pos, 0)?;
        if pos != bytes.len() {
            return Err(DecodeError::Trailing(pos));
        }
        if t.encode() != bytes {
            return Err(DecodeError::NotCanonical);
        }
        if !t.is_closed() {
            return Err(DecodeError::HoleOutsideContext);
        }
        Ok(t)
    }
}

impl PartialEq for Term {
    fn eq(&self, o: &Self) -> bool {
        Arc::ptr_eq(&self.0, &o.0) || self.encode() == o.encode()
    }
}
impl Eq for Term {}
impl fmt::Debug for Term {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", print::term_inline(self))
    }
}

// ---------------------------------------------------------------------------
// Filling a context (Req. 5.25)

/// `C[v]`: substitute `v` for every hole bound by `C`'s own context node,
/// hereditarily beneath quotation and through stores, and through nested
/// contexts at the shifted index. Results are rebuilt through the smart
/// constructors, so they are normal forms with fresh cached encodings. Only
/// hole-bearing subterms are visited; everything closed is reused.
pub fn fill(ctx: &Term, v: &CName) -> Term {
    let mut memo: HashMap<(Vec<u8>, u32), CName> = HashMap::new();
    fill_term(ctx, v, 0, &mut memo)
}

fn fill_term(t: &Term, v: &CName, depth: u32, memo: &mut HashMap<(Vec<u8>, u32), CName>) -> Term {
    if t.free_holes() <= depth {
        return t.clone();
    }
    Term::from_atoms(t.atoms().iter().map(|a| fill_atom(a, v, depth, memo)).collect())
}

fn fill_atom(a: &Atom, v: &CName, depth: u32, memo: &mut HashMap<(Vec<u8>, u32), CName>) -> Atom {
    if a.free_holes() <= depth {
        return a.clone();
    }
    let names: Vec<CName> = a.names().iter().map(|n| fill_name(n, v, depth, memo)).collect();
    match a.shape() {
        Shape::Q => Atom::build(Shape::Q, names, Some(fill_term(a.store().unwrap(), v, depth, memo))),
        Shape::Inst | Shape::Cstar => {
            Atom::build(a.shape(), names, Some(fill_term(a.context().unwrap(), v, depth + 1, memo)))
        }
        s => Atom::build(s, names, None),
    }
}

fn fill_name(n: &CName, v: &CName, depth: u32, memo: &mut HashMap<(Vec<u8>, u32), CName>) -> CName {
    if n.free_holes() <= depth {
        return n.clone();
    }
    match n.is_hole() {
        Some(k) if k == depth => v.clone(),
        Some(k) if k > depth => CName::hole(k - 1),
        Some(_) => n.clone(),
        None => {
            let key = (n.encode().to_vec(), depth);
            if let Some(x) = memo.get(&key) {
                return x.clone();
            }
            let r = CName::quote(fill_term(&n.drop(), v, depth, memo));
            memo.insert(key, r.clone());
            r
        }
    }
}

// ---------------------------------------------------------------------------
// Decoding

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DecodeError {
    Eof,
    /// A tag outside the target block, a source tag, the drop tag, or a
    /// context node / hole out of place.
    BadTag(u8, usize),
    Trailing(usize),
    NotCanonical,
    TooDeep,
    /// A hole not bound by a context node (`comb-context-misplaced`).
    HoleOutsideContext,
}

impl fmt::Display for DecodeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DecodeError::Eof => write!(f, "unexpected end of input [comb-decode]"),
            DecodeError::BadTag(t, p) => {
                let why = if *t <= 0x20 {
                    "a K1ndl1ng source tag where a target tag was expected"
                } else if *t == tags::DROP {
                    "the drop tag, which is unreachable under presentation A"
                } else if *t == tags::CONTEXT {
                    "a context node outside the third argument of inst [comb-context-misplaced]"
                } else {
                    "an unknown tag"
                };
                write!(f, "byte {p}: tag 0x{t:02x} is {why} [comb-decode]")
            }
            DecodeError::Trailing(p) => write!(f, "trailing bytes after offset {p} [comb-decode]"),
            DecodeError::NotCanonical => write!(f, "the encoding is not canonical [comb-decode]"),
            DecodeError::TooDeep => write!(f, "nesting exceeds the decoder's depth bound [comb-decode]"),
            DecodeError::HoleOutsideContext => {
                write!(f, "a hole outside any context node [comb-context-misplaced]")
            }
        }
    }
}

/// Quotation depth bound for the recursive decoder.
pub const MAX_DECODE_DEPTH: u32 = 4096;

fn decode_name(b: &[u8], pos: &mut usize, depth: u32) -> Result<CName, DecodeError> {
    let t = *b.get(*pos).ok_or(DecodeError::Eof)?;
    match t {
        tags::QUOTE => {
            *pos += 1;
            Ok(CName::quote(decode_term(b, pos, depth + 1)?))
        }
        tags::HOLE => {
            *pos += 1;
            let k = read_leb(b, pos).ok_or(DecodeError::Eof)?;
            Ok(CName::hole(k as u32))
        }
        _ => Err(DecodeError::BadTag(t, *pos)),
    }
}

fn decode_atom(b: &[u8], pos: &mut usize, depth: u32) -> Result<Atom, DecodeError> {
    let at = *pos;
    let t = *b.get(*pos).ok_or(DecodeError::Eof)?;
    let shape = Shape::from_tag(t).ok_or(DecodeError::BadTag(t, at))?;
    *pos += 1;
    let mut names = Vec::with_capacity(shape.arity());
    let mut aux = None;
    for j in 0..shape.arity() {
        match shape.arg(j) {
            Arg::Name => names.push(decode_name(b, pos, depth)?),
            Arg::Store => aux = Some(decode_term(b, pos, depth + 1)?),
            Arg::Context => {
                let c = *b.get(*pos).ok_or(DecodeError::Eof)?;
                if c != tags::CONTEXT {
                    return Err(DecodeError::BadTag(c, *pos));
                }
                *pos += 1;
                aux = Some(decode_term(b, pos, depth + 1)?);
            }
        }
    }
    Ok(Atom::build(shape, names, aux))
}

fn decode_term(b: &[u8], pos: &mut usize, depth: u32) -> Result<Term, DecodeError> {
    if depth > MAX_DECODE_DEPTH {
        return Err(DecodeError::TooDeep);
    }
    let t = *b.get(*pos).ok_or(DecodeError::Eof)?;
    match t {
        tags::NIL => {
            *pos += 1;
            Ok(Term::nil())
        }
        tags::PAR => {
            *pos += 1;
            let n = read_leb(b, pos).ok_or(DecodeError::Eof)? as usize;
            if n < 2 {
                return Err(DecodeError::NotCanonical);
            }
            let mut atoms = Vec::with_capacity(n.min(1 << 16));
            for _ in 0..n {
                atoms.push(decode_atom(b, pos, depth)?);
            }
            Ok(Term::from_atoms(atoms))
        }
        _ => Ok(Term::atom(decode_atom(b, pos, depth)?)),
    }
}

#[cfg(test)]
mod tests;
