//! `f1r3comb-mat` — the machine representation of F1R3Comb-Mat (v0.1 §3, as carried by v0.5), with templates over holes (v0.5 §3.3).
//!
//! This is a *machine representation*, not a term type (Req. 3.7): the public
//! term API is `f1r3comb-term`'s. A name is an interned [`NameId`]; a state is
//! one structure-of-arrays table per shape; `.cmat` is the canonical,
//! relabelled export consumed by device backends and golden tests.
//!
//! Ids never leave the process except through `.cmat`, which relabels them
//! canonically (Req. 3.3); externally a name is its content hash.

#![forbid(unsafe_code)]

use f1r3comb_term::{leb, read_leb, Arg, Atom, CName, Hash32, Shape, Term, SHAPES};
use std::collections::HashMap;

/// An interned name.
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct NameId(pub u32);

impl NameId {
    /// Filler for unused argument slots.
    pub const PAD: NameId = NameId(u32::MAX);
}

/// One component of a name, or one row of the state. For `q`, `args[1]` is
/// the id of the *quotation* of the stored process (Mat Decision 3.2).
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct Row {
    pub shape: Shape,
    pub args: [NameId; 4],
}

impl Row {
    pub fn new(shape: Shape, a: &[NameId]) -> Row {
        let mut args = [NameId::PAD; 4];
        args[..a.len()].copy_from_slice(a);
        Row { shape, args }
    }
    pub fn used(&self) -> &[NameId] {
        &self.args[..self.shape.arity()]
    }
}

/// Limits on the intern table (Req. 3.4; `comb-name-budget`).
#[derive(Copy, Clone, Debug)]
pub struct NameBudget {
    pub max_names: u32,
    pub max_components: u32,
    pub max_arena: u64,
}

impl Default for NameBudget {
    fn default() -> Self {
        NameBudget { max_names: 1 << 24, max_components: 1 << 20, max_arena: 1 << 28 }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum BudgetError {
    Names(u32),
    Components(u32),
    Arena(u64),
    /// A `cstar` template outside the curried validity class.
    Template,
}

impl std::fmt::Display for BudgetError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BudgetError::Names(n) => write!(f, "more than {n} distinct names [comb-name-budget]"),
            BudgetError::Components(n) => write!(f, "a name with more than {n} components [comb-name-budget]"),
            BudgetError::Arena(n) => write!(f, "arena longer than {n} rows [comb-name-budget]"),
            BudgetError::Template => write!(f, "a cstar template outside the curried validity class [comb-mat-template]"),
        }
    }
}

/// The bijection between ids and canonical component multisets
/// (Mat Def. 3.1). `quote` and `drop` are hash-free (Req. 3.2).
pub struct InternTable {
    arena: Vec<Row>,
    span: Vec<(u32, u32)>,
    key: HashMap<Box<[Row]>, NameId>,
    depth: Vec<u32>,
    chash: Vec<Option<Hash32>>,
    names: Vec<Option<CName>>,
    import_memo: HashMap<Hash32, NameId>,
    /// `Some(k)` for the hole of de Bruijn index `k`.
    hole: Vec<Option<u32>>,
    holes: HashMap<u32, NameId>,
    /// Context levels a name needs from outside (0 = closed).
    free: Vec<u32>,
    pub budget: NameBudget,
}

impl Default for InternTable {
    fn default() -> Self {
        InternTable::new(NameBudget::default())
    }
}

impl InternTable {
    pub fn new(budget: NameBudget) -> InternTable {
        InternTable {
            arena: Vec::new(),
            span: Vec::new(),
            key: HashMap::new(),
            depth: Vec::new(),
            chash: Vec::new(),
            names: Vec::new(),
            import_memo: HashMap::new(),
            hole: Vec::new(),
            holes: HashMap::new(),
            free: Vec::new(),
            budget,
        }
    }

    pub fn len(&self) -> usize {
        self.span.len()
    }

    pub fn is_empty(&self) -> bool {
        self.span.is_empty()
    }

    pub fn arena_len(&self) -> usize {
        self.arena.len()
    }

    /// Would `quote` of these rows need a new id?
    pub fn lookup(&self, rows: &[Row]) -> Option<NameId> {
        self.key.get(rows).copied()
    }

    /// Insert-or-get by structural key. `rows` must be sorted.
    pub fn quote(&mut self, rows: &[Row]) -> Result<NameId, BudgetError> {
        debug_assert!(rows.windows(2).all(|w| w[0] <= w[1]), "unsorted key");
        if let Some(id) = self.key.get(rows) {
            return Ok(*id);
        }
        if self.span.len() as u64 >= self.budget.max_names as u64 {
            return Err(BudgetError::Names(self.budget.max_names));
        }
        if rows.len() as u64 > self.budget.max_components as u64 {
            return Err(BudgetError::Components(self.budget.max_components));
        }
        if (self.arena.len() + rows.len()) as u64 > self.budget.max_arena {
            return Err(BudgetError::Arena(self.budget.max_arena));
        }
        let id = NameId(self.span.len() as u32);
        let off = self.arena.len() as u32;
        self.arena.extend_from_slice(rows);
        self.span.push((off, rows.len() as u32));
        let mut d = 0u32;
        let mut fr = 0u32;
        for r in rows {
            for (j, a) in r.used().iter().enumerate() {
                d = d.max(self.depth[a.0 as usize] + 1);
                let f = self.free[a.0 as usize];
                fr = fr.max(if r.shape.arg(j) == Arg::Context { f.saturating_sub(1) } else { f });
            }
        }
        self.depth.push(d);
        self.chash.push(None);
        self.names.push(None);
        self.hole.push(None);
        self.free.push(fr);
        self.key.insert(rows.to_vec().into_boxed_slice(), id);
        Ok(id)
    }

    /// The components of a name: an arena read.
    pub fn drop(&self, id: NameId) -> &[Row] {
        let (o, l) = self.span[id.0 as usize];
        &self.arena[o as usize..(o + l) as usize]
    }

    pub fn depth(&self, id: NameId) -> u32 {
        self.depth[id.0 as usize]
    }

    /// The hole of de Bruijn index `k` (a template's parameter,
    /// Mat v0.5 Req. 3.3, with indices kept: see the README).
    pub fn hole(&mut self, k: u32) -> NameId {
        if let Some(id) = self.holes.get(&k) {
            return *id;
        }
        let id = NameId(self.span.len() as u32);
        self.span.push((self.arena.len() as u32, 0));
        self.depth.push(0);
        self.chash.push(None);
        self.names.push(Some(CName::hole(k)));
        self.hole.push(Some(k));
        self.free.push(k + 1);
        self.holes.insert(k, id);
        id
    }

    pub fn is_hole(&self, id: NameId) -> Option<u32> {
        self.hole[id.0 as usize]
    }

    /// Context levels a name needs from outside.
    pub fn free(&self, id: NameId) -> u32 {
        self.free[id.0 as usize]
    }

    /// `T[v]` over the interned form (Mat Req. 3.5): substitute `v` for the
    /// holes `T`'s own context binds, hereditarily beneath quotation and
    /// through nested contexts at the shifted index. Visits only names that
    /// mention a hole (the precomputed `free` column); everything else is
    /// reused. New names are interned in a fixed traversal order. Returns
    /// the result and the number of names built.
    pub fn fill(&mut self, t: NameId, v: NameId) -> Result<(NameId, u64), BudgetError> {
        let before = self.len();
        let mut memo: HashMap<(NameId, u32), NameId> = HashMap::new();
        let r = self.fill_at(t, v, 0, &mut memo)?;
        Ok((r, (self.len() - before) as u64))
    }

    fn fill_at(&mut self, n: NameId, v: NameId, depth: u32, memo: &mut HashMap<(NameId, u32), NameId>) -> Result<NameId, BudgetError> {
        if self.free(n) <= depth {
            return Ok(n);
        }
        if let Some(k) = self.is_hole(n) {
            return Ok(if k == depth { v } else if k > depth { self.hole(k - 1) } else { n });
        }
        if let Some(x) = memo.get(&(n, depth)) {
            return Ok(*x);
        }
        let rows: Vec<Row> = self.drop(n).to_vec();
        let mut out = Vec::with_capacity(rows.len());
        for r in rows {
            let mut args = r.args;
            for j in 0..r.shape.arity() {
                let dj = if r.shape.arg(j) == Arg::Context { depth + 1 } else { depth };
                args[j] = self.fill_at(r.args[j], v, dj, memo)?;
            }
            out.push(Row { shape: r.shape, args });
        }
        out.sort();
        let id = self.quote(&out)?;
        memo.insert((n, depth), id);
        Ok(id)
    }

    /// Intern a target name, bottom-up with an explicit stack (Req. 3.8).
    pub fn intern(&mut self, n: &CName) -> Result<NameId, BudgetError> {
        let h = n.content_hash();
        if let Some(id) = self.import_memo.get(&h) {
            return Ok(*id);
        }
        // Frames: (name, children pushed?)
        let mut stack: Vec<(CName, bool)> = vec![(n.clone(), false)];
        let mut done: HashMap<Hash32, NameId> = HashMap::new();
        while let Some((cur, expanded)) = stack.pop() {
            let ch = cur.content_hash();
            if done.contains_key(&ch) || self.import_memo.contains_key(&ch) {
                continue;
            }
            if let Some(k) = cur.is_hole() {
                let id = self.hole(k);
                done.insert(ch, id);
                self.import_memo.insert(ch, id);
                continue;
            }
            let proc_ = cur.drop();
            if !expanded {
                stack.push((cur.clone(), true));
                for a in proc_.atoms() {
                    for x in a.names() {
                        stack.push((x.clone(), false));
                    }
                    if let Some(p) = a.store().or(a.context()) {
                        stack.push((CName::quote(p.clone()), false));
                    }
                }
                continue;
            }
            let get = |x: &CName, done: &HashMap<Hash32, NameId>, memo: &HashMap<Hash32, NameId>| {
                let h = x.content_hash();
                *done.get(&h).or_else(|| memo.get(&h)).expect("child interned")
            };
            let mut rows = Vec::with_capacity(proc_.len());
            for a in proc_.atoms() {
                let mut ids: Vec<NameId> =
                    a.names().iter().map(|x| get(x, &done, &self.import_memo)).collect();
                if let Some(p) = a.store().or(a.context()) {
                    ids.push(get(&CName::quote(p.clone()), &done, &self.import_memo));
                }
                rows.push(Row::new(a.shape(), &ids));
            }
            rows.sort();
            let id = self.quote(&rows)?;
            self.chash[id.0 as usize] = Some(ch);
            self.names[id.0 as usize] = Some(cur.clone());
            done.insert(ch, id);
            self.import_memo.insert(ch, id);
        }
        Ok(self.import_memo[&h])
    }

    /// The target name of an id (export boundary), memoised, explicit stack.
    pub fn name(&mut self, id: NameId) -> CName {
        if let Some(n) = &self.names[id.0 as usize] {
            return n.clone();
        }
        let mut stack = vec![(id, false)];
        while let Some((cur, expanded)) = stack.pop() {
            if self.names[cur.0 as usize].is_some() {
                continue;
            }
            let rows: Vec<Row> = self.drop(cur).to_vec();
            if !expanded {
                stack.push((cur, true));
                for r in &rows {
                    for a in r.used() {
                        if self.names[a.0 as usize].is_none() {
                            stack.push((*a, false));
                        }
                    }
                }
                continue;
            }
            let atoms: Vec<Atom> = rows.iter().map(|r| self.row_atom_cached(r)).collect();
            let n = CName::quote(Term::from_atoms(atoms));
            self.names[cur.0 as usize] = Some(n);
        }
        self.names[id.0 as usize].clone().unwrap()
    }

    fn row_atom_cached(&self, r: &Row) -> Atom {
        let g = |x: NameId| self.names[x.0 as usize].clone().expect("child named");
        if r.shape == Shape::Q {
            Atom::q(g(r.args[0]), g(r.args[1]).drop())
        } else if matches!(r.shape, Shape::Inst) {
            Atom::inst(g(r.args[0]), g(r.args[1]), g(r.args[2]).drop())
        } else if matches!(r.shape, Shape::Cstar) {
            Atom::cstar(g(r.args[0]), g(r.args[1]), g(r.args[2]).drop())
        } else {
            Atom::new(r.shape, r.used().iter().map(|x| g(*x)).collect()).expect("arity")
        }
    }

    /// The atom a state row denotes.
    pub fn row_atom(&mut self, r: &Row) -> Atom {
        for a in r.used() {
            self.name(*a);
        }
        self.row_atom_cached(r)
    }

    /// Content hash of a name (boundary only; memoised).
    pub fn chash(&mut self, id: NameId) -> Hash32 {
        if let Some(h) = self.chash[id.0 as usize] {
            return h;
        }
        let h = self.name(id).content_hash();
        self.chash[id.0 as usize] = Some(h);
        h
    }
}

/// One shape's table: `arity` columns of equal length.
#[derive(Clone, Default, Debug, PartialEq, Eq)]
pub struct ShapeTable {
    pub cols: [Vec<NameId>; 4],
    pub len: u32,
}

impl ShapeTable {
    pub fn row(&self, shape: Shape, i: usize) -> Row {
        let mut args = [NameId::PAD; 4];
        for (j, a) in args.iter_mut().enumerate().take(shape.arity()) {
            *a = self.cols[j][i];
        }
        Row { shape, args }
    }
    pub fn push(&mut self, r: &Row) {
        for j in 0..r.shape.arity() {
            self.cols[j].push(r.args[j]);
        }
        self.len += 1;
    }
}

/// A state: one table per shape (Mat Def. 3.5), structure of arrays.
#[derive(Clone, Default, Debug, PartialEq, Eq)]
pub struct State {
    pub tables: [ShapeTable; SHAPES],
}

impl State {
    pub fn table(&self, s: Shape) -> &ShapeTable {
        &self.tables[s.ix()]
    }

    pub fn rows(&self) -> u64 {
        self.tables.iter().map(|t| t.len as u64).sum()
    }

    pub fn push(&mut self, r: &Row) {
        self.tables[r.shape.ix()].push(r);
    }

    /// Flatten a term's components in encoding order, one row per atom.
    pub fn import(t: &Term, it: &mut InternTable) -> Result<State, BudgetError> {
        let mut st = State::default();
        for a in t.atoms() {
            let mut ids = Vec::with_capacity(4);
            for n in a.names() {
                ids.push(it.intern(n)?);
            }
            if let Some(p) = a.store().or(a.context()) {
                ids.push(it.intern(&CName::quote(p.clone()))?);
            }
            st.push(&Row::new(a.shape(), &ids));
        }
        Ok(st)
    }

    /// Rebuild the term (Req. 3.9).
    pub fn export(&self, it: &mut InternTable) -> Term {
        let mut atoms = Vec::with_capacity(self.rows() as usize);
        for s in Shape::ALL {
            let t = self.table(s);
            for i in 0..t.len as usize {
                atoms.push(it.row_atom(&t.row(s, i)));
            }
        }
        Term::from_atoms(atoms)
    }

    /// All rows, shape by shape.
    pub fn all_rows(&self) -> Vec<Row> {
        let mut v = Vec::new();
        for s in Shape::ALL {
            let t = self.table(s);
            for i in 0..t.len as usize {
                v.push(t.row(s, i));
            }
        }
        v
    }
}

// ---------------------------------------------------------------------------
// .cmat

pub const CMAT_MAGIC: &[u8; 4] = b"CMAT";
pub const CMAT_VERSION: u8 = 2;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CmatError {
    Magic,
    Version(u8),
    Presentation(u8),
    Feature(u8),
    Shapes(u8),
    Truncated,
    BadTag(u8),
    ForwardRef,
    NotCanonical,
    Budget(BudgetError),
}

impl std::fmt::Display for CmatError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CmatError::Magic => write!(f, "not a .cmat file [comb-decode]"),
            CmatError::Version(v) => write!(f, "unsupported .cmat version {v} [comb-decode]"),
            CmatError::Presentation(p) => {
                write!(f, "presentation {:?}, this build is A [comb-presentation-mismatch]", *p as char)
            }
            CmatError::Feature(s) => write!(f, "shortcuts={s}, this build has 0 [comb-feature-mismatch]"),
            CmatError::Shapes(n) => write!(f, "{n} shapes, this build has {SHAPES} [comb-feature-mismatch]"),
            CmatError::Truncated => write!(f, "truncated .cmat [comb-decode]"),
            CmatError::BadTag(t) => write!(f, "bad shape tag 0x{t:02x} [comb-decode]"),
            CmatError::ForwardRef => write!(f, "a name refers to a later id [comb-decode]"),
            CmatError::NotCanonical => write!(f, "the file is not in canonical order [comb-decode]"),
            CmatError::Budget(b) => write!(f, "{b}"),
        }
    }
}

/// The canonical relabelling (Mat Def. 3.10) of every name reachable from
/// `st`: order by quotation depth, ties by content hash.
pub fn relabel(st: &State, it: &mut InternTable) -> Vec<NameId> {
    let mut seen = vec![false; it.len()];
    let mut order = Vec::new();
    let mut stack: Vec<NameId> = Vec::new();
    for r in st.all_rows() {
        stack.extend_from_slice(r.used());
    }
    while let Some(x) = stack.pop() {
        if seen[x.0 as usize] {
            continue;
        }
        seen[x.0 as usize] = true;
        order.push(x);
        for r in it.drop(x) {
            stack.extend_from_slice(r.used());
        }
    }
    let mut keyed: Vec<(u32, Hash32, NameId)> =
        order.into_iter().map(|x| (it.depth(x), it.chash(x), x)).collect();
    keyed.sort_by(|a, b| (a.0, a.1 .0).cmp(&(b.0, b.1 .0)));
    keyed.into_iter().map(|k| k.2).collect()
}

/// Encode a state as `.cmat` (Mat Req. 3.11).
pub fn cmat_encode(st: &State, it: &mut InternTable) -> Vec<u8> {
    let order = relabel(st, it);
    let mut map: HashMap<NameId, u32> = HashMap::with_capacity(order.len());
    for (i, x) in order.iter().enumerate() {
        map.insert(*x, i as u32);
    }
    let mut out = Vec::new();
    out.extend_from_slice(CMAT_MAGIC);
    out.push(CMAT_VERSION);
    out.push(f1r3comb_term::artefact::PRESENTATION);
    out.push(f1r3comb_term::artefact::SHORTCUTS);
    out.push(SHAPES as u8);
    leb(order.len() as u64, &mut out);
    for x in &order {
        if let Some(k) = it.is_hole(*x) {
            leb(1, &mut out);
            leb(k as u64, &mut out);
            continue;
        }
        let mut rows: Vec<(u8, Vec<u32>)> = it
            .drop(*x)
            .iter()
            .map(|r| (r.shape.tag(), r.used().iter().map(|a| map[a]).collect()))
            .collect();
        rows.sort();
        leb(rows.len() as u64 * 2, &mut out);
        for (tag, ids) in rows {
            out.push(tag);
            for i in ids {
                leb(i as u64, &mut out);
            }
        }
    }
    for s in Shape::ALL {
        let t = st.table(s);
        let mut rows: Vec<Vec<u32>> = (0..t.len as usize)
            .map(|i| (0..s.arity()).map(|j| map[&t.cols[j][i]]).collect())
            .collect();
        rows.sort();
        leb(rows.len() as u64, &mut out);
        for j in 0..s.arity() {
            for r in &rows {
                leb(r[j] as u64, &mut out);
            }
        }
    }
    out
}

/// Decode a `.cmat` into a fresh intern table and a state whose ids are the
/// file's canonical ids.
pub fn cmat_decode(b: &[u8], budget: NameBudget) -> Result<(State, InternTable), CmatError> {
    if b.len() < 8 || &b[..4] != CMAT_MAGIC {
        return Err(CmatError::Magic);
    }
    if b[4] != CMAT_VERSION {
        return Err(CmatError::Version(b[4]));
    }
    if b[5] != f1r3comb_term::artefact::PRESENTATION {
        return Err(CmatError::Presentation(b[5]));
    }
    if b[6] != f1r3comb_term::artefact::SHORTCUTS {
        return Err(CmatError::Feature(b[6]));
    }
    if b[7] as usize != SHAPES {
        return Err(CmatError::Shapes(b[7]));
    }
    let mut pos = 8usize;
    let rd = |pos: &mut usize| read_leb(b, pos).ok_or(CmatError::Truncated);
    let n = rd(&mut pos)?;
    let mut it = InternTable::new(budget);
    for i in 0..n {
        let c2 = rd(&mut pos)?;
        if c2 & 1 == 1 {
            let k = rd(&mut pos)? as u32;
            let id = it.hole(k);
            if id.0 as u64 != i {
                return Err(CmatError::NotCanonical);
            }
            continue;
        }
        let c = c2 / 2;
        let mut rows = Vec::with_capacity(c.min(1 << 16) as usize);
        for _ in 0..c {
            let tag = *b.get(pos).ok_or(CmatError::Truncated)?;
            pos += 1;
            let shape = Shape::from_tag(tag).ok_or(CmatError::BadTag(tag))?;
            let mut ids = Vec::with_capacity(4);
            for _ in 0..shape.arity() {
                let x = rd(&mut pos)?;
                if x >= i {
                    return Err(CmatError::ForwardRef);
                }
                ids.push(NameId(x as u32));
            }
            rows.push(Row::new(shape, &ids));
        }
        rows.sort();
        let id = it.quote(&rows).map_err(CmatError::Budget)?;
        if id.0 as u64 != i {
            return Err(CmatError::NotCanonical); // duplicate name
        }
    }
    let mut st = State::default();
    for s in Shape::ALL {
        let len = rd(&mut pos)? as usize;
        let mut cols: Vec<Vec<NameId>> = vec![Vec::with_capacity(len.min(1 << 20)); s.arity()];
        for col in cols.iter_mut() {
            for _ in 0..len {
                let x = rd(&mut pos)?;
                if x >= n {
                    return Err(CmatError::ForwardRef);
                }
                col.push(NameId(x as u32));
            }
        }
        let t = &mut st.tables[s.ix()];
        for (j, col) in cols.into_iter().enumerate() {
            t.cols[j] = col;
        }
        t.len = len as u32;
    }
    if pos != b.len() {
        return Err(CmatError::Truncated);
    }
    // Canonicity: re-encoding must reproduce the bytes.
    if cmat_encode(&st, &mut it) != b {
        return Err(CmatError::NotCanonical);
    }
    Ok((st, it))
}

#[cfg(test)]
mod tests;
