//! `f1r3comb-par` — the host machine of F1R3Comb-Mat (v0.1 §5–§9, as carried by v0.5), running the generated rule table including `inst` and `cstar`.
//!
//! One step: find every enabled redex (sorted by canonical key), give each a
//! priority from the run seed, the step number and its key, take the greedy
//! maximal independent set in priority order, and commit it in canonical
//! order. The step is a function of the state, the seed and the
//! configuration (Mat Prop. 7.2), which is what lets a device backend be
//! checked against this one step for step.
//!
//! The device backend (`f1r3comb-gpu`) reuses [`find_bucketed`]'s ordering,
//! [`priority`], the rounds formulation [`select_rounds`] and [`commit`].

#![forbid(unsafe_code)]

use f1r3comb_mat::{BudgetError, InternTable, NameId, Row, State};
use f1r3comb_term::rules::{rules, Conclusion, Datum, NodeTemplate, Var, RULE_COUNT};
use f1r3comb_term::addr::{largest_name, r_star, Allocator};
use f1r3comb_term::{Atom, CName, Hash32, Shape, Term};

/// Deploy a compiled term (F1R3Comb v0.6 Req. 9.6): if its top level holds a
/// unit waiting for an address, post a fresh root address — larger than
/// every name of the term, prefix-incomparable with every root the
/// allocator has issued — at `r*`.
pub fn deploy(t: &Term, alloc: &mut Allocator) -> (Term, Option<CName>) {
    let waits = t.atoms().iter().any(|a| a.shape() == Shape::Inst && *a.subject() == r_star());
    if !waits {
        return (t.clone(), None);
    }
    let root = alloc.root(&largest_name(t));
    let mut atoms = t.atoms().to_vec();
    atoms.push(Atom::m(r_star(), root.clone()));
    (Term::from_atoms(atoms), Some(root))
}

/// A redex (Mat Def. 5.1). Field order is the canonical key (Def. 5.2).
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct Redex {
    pub rule: u8,
    pub consumer: u32,
    /// Premise rows; `u32::MAX` beyond the rule's premise count.
    pub prem: [u32; 3],
}

pub const PAD: u32 = u32::MAX;

fn datum_shape(d: Datum) -> Shape {
    match d {
        Datum::Loud => Shape::M,
        Datum::Quiet => Shape::Q,
    }
}

// ---------------------------------------------------------------------------
// Buckets (Req. 5.3)

/// For each name occurring in a column, the ascending rows carrying it; CSR
/// over the occurring names only.
#[derive(Clone, Debug, Default)]
pub struct Buckets {
    pub names: Vec<NameId>,
    pub indptr: Vec<u32>,
    pub rows: Vec<u32>,
}

impl Buckets {
    pub fn of(col: &[NameId]) -> Buckets {
        let mut pairs: Vec<(NameId, u32)> = col.iter().enumerate().map(|(i, n)| (*n, i as u32)).collect();
        pairs.sort_unstable();
        let mut b = Buckets { indptr: vec![0], ..Default::default() };
        for (n, r) in pairs {
            if b.names.last() != Some(&n) {
                if !b.names.is_empty() {
                    b.indptr.push(b.rows.len() as u32);
                }
                b.names.push(n);
            }
            b.rows.push(r);
        }
        if !b.names.is_empty() {
            b.indptr.push(b.rows.len() as u32);
        }
        b
    }

    pub fn get(&self, n: NameId) -> &[u32] {
        match self.names.binary_search(&n) {
            Ok(i) => &self.rows[self.indptr[i] as usize..self.indptr[i + 1] as usize],
            Err(_) => &[],
        }
    }
}

// ---------------------------------------------------------------------------
// Finders (Req. 5.4)

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Finder {
    /// Nested loops over whole premise tables: the oracle.
    Naive,
    /// Bucket lookups: the production finder.
    Bucketed,
}

/// Premise atoms equal in content (subject and payload)?
fn same_atom(st: &State, s: Shape, a: u32, b: u32) -> bool {
    let t = st.table(s);
    t.cols[0][a as usize] == t.cols[0][b as usize] && t.cols[1][a as usize] == t.cols[1][b as usize]
}

/// Distinctness and symmetry masks (Mat Def. 5.1, Req. 5.3a).
fn admissible(st: &State, rule: usize, p: &[u32]) -> bool {
    let prem = &rules()[rule].premises;
    for i in 0..p.len() {
        for j in i + 1..p.len() {
            if prem[i].datum == prem[j].datum {
                let s = datum_shape(prem[i].datum);
                if p[i] == p[j] {
                    return false;
                }
                if p[j] < p[i] && same_atom(st, s, p[i], p[j]) {
                    return false;
                }
            }
        }
    }
    true
}

pub fn find(st: &State, f: Finder) -> Vec<Redex> {
    match f {
        Finder::Naive => find_naive(st),
        Finder::Bucketed => find_bucketed(st),
    }
}

pub fn find_naive(st: &State) -> Vec<Redex> {
    let mut out = Vec::new();
    for (r, spec) in rules().iter().enumerate() {
        let ct = st.table(spec.consumer);
        for c in 0..ct.len {
            let mut cur: Vec<u32> = Vec::new();
            naive_rec(st, r, c, 0, &mut cur, &mut out);
        }
    }
    out.sort_unstable();
    out
}

fn naive_rec(st: &State, r: usize, c: u32, k: usize, cur: &mut Vec<u32>, out: &mut Vec<Redex>) {
    let spec = &rules()[r];
    if k == spec.premises.len() {
        if admissible(st, r, cur) {
            let mut prem = [PAD; 3];
            prem[..cur.len()].copy_from_slice(cur);
            out.push(Redex { rule: r as u8, consumer: c, prem });
        }
        return;
    }
    let p = spec.premises[k];
    let want = st.table(spec.consumer).cols[p.subject as usize][c as usize];
    let t = st.table(datum_shape(p.datum));
    for i in 0..t.len {
        if t.cols[0][i as usize] == want {
            cur.push(i);
            naive_rec(st, r, c, k + 1, cur, out);
            cur.pop();
        }
    }
}

/// Generated in canonical order: rules, consumer rows ascending, bucket
/// entries ascending (Req. 5.5) — no sort.
pub fn find_bucketed(st: &State) -> Vec<Redex> {
    let bm = Buckets::of(&st.table(Shape::M).cols[0]);
    let bq = Buckets::of(&st.table(Shape::Q).cols[0]);
    let mut out = Vec::new();
    for (r, spec) in rules().iter().enumerate() {
        let ct = st.table(spec.consumer);
        for c in 0..ct.len as usize {
            let lists: Vec<&[u32]> = spec
                .premises
                .iter()
                .map(|p| {
                    let n = ct.cols[p.subject as usize][c];
                    match p.datum {
                        Datum::Loud => bm.get(n),
                        Datum::Quiet => bq.get(n),
                    }
                })
                .collect();
            if lists.iter().any(|l| l.is_empty()) {
                continue;
            }
            let mut idx = vec![0usize; lists.len()];
            'outer: loop {
                let cur: Vec<u32> = idx.iter().zip(&lists).map(|(i, l)| l[*i]).collect();
                if admissible(st, r, &cur) {
                    let mut prem = [PAD; 3];
                    prem[..cur.len()].copy_from_slice(&cur);
                    out.push(Redex { rule: r as u8, consumer: c as u32, prem });
                }
                // odometer, last position fastest
                let mut k = lists.len();
                loop {
                    if k == 0 {
                        break 'outer;
                    }
                    k -= 1;
                    idx[k] += 1;
                    if idx[k] < lists[k].len() {
                        break;
                    }
                    idx[k] = 0;
                }
            }
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Priority (Mat Def. 6.2)

pub fn mix64(mut z: u64) -> u64 {
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
    z ^ (z >> 31)
}

pub fn priority(seed: u64, step: u64, r: &Redex) -> u64 {
    let mut h = mix64(mix64(mix64(seed) ^ step) ^ r.rule as u64);
    h = mix64(h ^ r.consumer as u64);
    for i in 0..rules()[r.rule as usize].premises.len() {
        h = mix64(h ^ r.prem[i] as u64);
    }
    h
}

/// The row references a redex consumes: `(table, row)` packed.
pub fn row_refs(r: &Redex) -> impl Iterator<Item = (Shape, u32)> + '_ {
    let spec = &rules()[r.rule as usize];
    std::iter::once((spec.consumer, r.consumer))
        .chain(spec.premises.iter().enumerate().map(move |(i, p)| (datum_shape(p.datum), r.prem[i])))
}

/// Indices of `rs` (sorted by key) in priority order.
pub fn priority_order(rs: &[Redex], seed: u64, step: u64) -> Vec<usize> {
    let mut v: Vec<(u64, usize)> = rs.iter().enumerate().map(|(i, r)| (priority(seed, step, r), i)).collect();
    v.sort_unstable();
    v.into_iter().map(|x| x.1).collect()
}

struct Taken(Vec<Vec<bool>>);

impl Taken {
    fn new(st: &State) -> Taken {
        Taken(Shape::ALL.iter().map(|s| vec![false; st.table(*s).len as usize]).collect())
    }
    fn get(&self, s: Shape, i: u32) -> bool {
        self.0[s.ix()][i as usize]
    }
    fn set(&mut self, s: Shape, i: u32) {
        self.0[s.ix()][i as usize] = true;
    }
}

/// The greedy maximal independent set (Def. 6.3). Returns chosen indices
/// into `rs`, and for each chosen redex the number it excluded.
pub fn select_greedy(st: &State, rs: &[Redex], order: &[usize]) -> Vec<usize> {
    let mut taken = Taken::new(st);
    let mut out = Vec::new();
    for &i in order {
        if row_refs(&rs[i]).all(|(s, x)| !taken.get(s, x)) {
            for (s, x) in row_refs(&rs[i]) {
                taken.set(s, x);
            }
            out.push(i);
        }
    }
    out.sort_unstable();
    out
}

/// The rounds formulation (Prop. 6.4) — what a device runs. Returns chosen
/// indices (sorted) and the number of rounds.
pub fn select_rounds(st: &State, rs: &[Redex], seed: u64, step: u64) -> (Vec<usize>, u32) {
    let rank: Vec<(u64, usize)> = rs.iter().enumerate().map(|(i, r)| (priority(seed, step, r), i)).collect();
    #[derive(Copy, Clone, PartialEq)]
    enum D {
        Undecided,
        In,
        Out,
    }
    let mut dec = vec![D::Undecided; rs.len()];
    let mut rounds = 0;
    let mut best: Vec<Vec<(u64, usize)>> =
        Shape::ALL.iter().map(|s| vec![(u64::MAX, usize::MAX); st.table(*s).len as usize]).collect();
    loop {
        if !dec.contains(&D::Undecided) {
            break;
        }
        rounds += 1;
        for b in best.iter_mut() {
            b.iter_mut().for_each(|x| *x = (u64::MAX, usize::MAX));
        }
        for (i, r) in rs.iter().enumerate() {
            if dec[i] == D::Undecided {
                for (s, x) in row_refs(r) {
                    let e = &mut best[s.ix()][x as usize];
                    *e = (*e).min(rank[i]);
                }
            }
        }
        let mut taken = Taken::new(st);
        for (i, r) in rs.iter().enumerate() {
            if dec[i] == D::Undecided && row_refs(r).all(|(s, x)| best[s.ix()][x as usize].1 == i) {
                dec[i] = D::In;
                for (s, x) in row_refs(r) {
                    taken.set(s, x);
                }
            }
        }
        for (i, r) in rs.iter().enumerate() {
            if dec[i] == D::Undecided && row_refs(r).any(|(s, x)| taken.get(s, x)) {
                dec[i] = D::Out;
            }
        }
    }
    ((0..rs.len()).filter(|i| dec[*i] == D::In).collect(), rounds)
}

// ---------------------------------------------------------------------------
// Firing (Mat Req. 7.1)

fn premise_payload(st: &State, spec_rule: usize, r: &Redex, j: usize) -> NameId {
    let p = rules()[spec_rule].premises[j];
    st.table(datum_shape(p.datum)).cols[1][r.prem[j] as usize]
}

fn var(st: &State, r: &Redex, v: Var) -> NameId {
    let spec = &rules()[r.rule as usize];
    match v {
        Var::Slot(j) => st.table(spec.consumer).cols[j as usize][r.consumer as usize],
        Var::Payload(j) => premise_payload(st, r.rule as usize, r, j as usize),
    }
}

/// Commit a set of redexes (indices into `rs`, which is sorted by key). A
/// budget error leaves the state untouched (Req. 7.3).
pub fn commit(st: &mut State, it: &mut InternTable, rs: &[Redex], chosen: &[usize]) -> Result<(), BudgetError> {
    let mut n = 0;
    commit_counting(st, it, rs, chosen, &mut n)
}

/// The curried validity class (Mat Req. 3.2): one atom, each argument a
/// path beneath the hole or a closed name.
pub fn curried_valid(it: &mut InternTable, t: NameId) -> bool {
    let rows = it.drop(t).to_vec();
    if rows.len() != 1 {
        return false;
    }
    rows[0].used().iter().all(|a| {
        if it.free(*a) == 0 {
            return true;
        }
        let (base, _) = f1r3comb_term::addr::spine(&it.name(*a));
        base.is_hole() == Some(0)
    })
}

/// [`commit`], also counting the names the step interned (the work figure
/// of Mat Req. 3.5, E6).
pub fn commit_counting(st: &mut State, it: &mut InternTable, rs: &[Redex], chosen: &[usize], names_built: &mut u64) -> Result<(), BudgetError> {
    let mut sel: Vec<Redex> = chosen.iter().map(|i| rs[*i]).collect();
    sel.sort_unstable();
    // step 2: conclusions, in canonical order
    let mut produced: Vec<Row> = Vec::new();
    for r in &sel {
        let spec = &rules()[r.rule as usize];
        match &spec.conclusion {
            Conclusion::Atoms(atoms) => {
                for (s, vars) in atoms {
                    let ids: Vec<NameId> = vars.iter().map(|v| var(st, r, *v)).collect();
                    produced.push(Row::new(*s, &ids));
                }
            }
            Conclusion::Decode { payload } => {
                let v = var(st, r, *payload);
                produced.extend_from_slice(it.drop(v));
            }
            Conclusion::Encode { deliver, node: NodeTemplate::Instantiate { template, with } } => {
                let t = var(st, r, *template);
                if spec.consumer == Shape::Cstar && !curried_valid(it, t) {
                    return Err(BudgetError::Template);
                }
                let (n, built) = it.fill(t, var(st, r, *with))?;
                *names_built += built;
                produced.push(Row::new(Shape::M, &[var(st, r, *deliver), n]));
            }
            Conclusion::Encode { deliver, node } => {
                let rows: Vec<Row> = match node {
                    NodeTemplate::Instantiate { .. } => unreachable!(),
                    NodeTemplate::Union(a, b) => {
                        let (x, y) = (var(st, r, *a), var(st, r, *b));
                        let mut m = Vec::with_capacity(it.drop(x).len() + it.drop(y).len());
                        let (xa, ya) = (it.drop(x), it.drop(y));
                        let (mut i, mut j) = (0, 0);
                        while i < xa.len() || j < ya.len() {
                            if j == ya.len() || (i < xa.len() && xa[i] <= ya[j]) {
                                m.push(xa[i]);
                                i += 1;
                            } else {
                                m.push(ya[j]);
                                j += 1;
                            }
                        }
                        m
                    }
                    NodeTemplate::Atom(s, vars) => {
                        let ids: Vec<NameId> = vars.iter().map(|v| var(st, r, *v)).collect();
                        vec![Row::new(*s, &ids)]
                    }
                };
                let before = it.len();
                let n = it.quote(&rows)?;
                *names_built += (it.len() - before) as u64;
                produced.push(Row::new(Shape::M, &[var(st, r, *deliver), n]));
            }
        }
    }
    // step 3: stable compaction of consumed rows
    let mut dead: Vec<Vec<bool>> = Shape::ALL.iter().map(|s| vec![false; st.table(*s).len as usize]).collect();
    for r in &sel {
        for (s, x) in row_refs(r) {
            dead[s.ix()][x as usize] = true;
        }
    }
    for s in Shape::ALL {
        let t = &mut st.tables[s.ix()];
        let kill = &dead[s.ix()];
        if !kill.iter().any(|k| *k) {
            continue;
        }
        for j in 0..s.arity() {
            let mut i = 0usize;
            t.cols[j].retain(|_| {
                let keep = !kill[i];
                i += 1;
                keep
            });
        }
        t.len = t.cols[0].len() as u32;
    }
    // step 4: append
    for row in &produced {
        st.push(row);
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Resolvers and the run loop

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Resolver {
    /// The greedy maximal step; the default.
    MaximalProgress,
    /// Fire only the first redex in priority order.
    SingleByPriority,
    /// Fire one redex chosen by `mix64` over the enumerated set.
    SingleUniform,
}

impl Resolver {
    pub fn name(self) -> &'static str {
        match self {
            Resolver::MaximalProgress => "maximal-progress",
            Resolver::SingleByPriority => "single-by-priority",
            Resolver::SingleUniform => "single-uniform",
        }
    }
    pub fn parse(s: &str) -> Option<Resolver> {
        [Resolver::MaximalProgress, Resolver::SingleByPriority, Resolver::SingleUniform]
            .into_iter()
            .find(|r| r.name() == s)
    }
}

#[derive(Clone, Debug)]
pub struct Config {
    pub seed: u64,
    pub resolver: Resolver,
    pub finder: Finder,
    pub max_steps: u64,
    /// Upper bound on live rows; exceeding it halts before commit.
    pub max_rows: u64,
    /// Record the content hash of the state after every step.
    pub hash_states: bool,
}

impl Default for Config {
    fn default() -> Self {
        Config {
            seed: 0,
            resolver: Resolver::MaximalProgress,
            finder: Finder::Bucketed,
            max_steps: 100_000,
            max_rows: 1 << 26,
            hash_states: false,
        }
    }
}

/// One step's record. Contains no ids or row positions (Req. 3.3).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StepRecord {
    pub step: u64,
    pub enumerated: u64,
    pub fired: u64,
    pub per_rule: [u64; RULE_COUNT],
    /// Names interned by the step's conclusions (the work figure).
    pub names_built: u64,
    pub state_hash: Option<Hash32>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Stop {
    /// No enabled redex.
    Quiescent,
    MaxSteps,
    Budget(String),
}

#[derive(Clone, Debug)]
pub struct RunReport {
    pub steps: Vec<StepRecord>,
    pub stop: Stop,
    pub fired: u64,
    pub final_term: Term,
}

impl RunReport {
    pub fn final_hash(&self) -> Hash32 {
        self.final_term.content_hash()
    }
}

/// Something that picks the step's redex set. The host does it here; the
/// device backend supplies its own.
pub trait StepEngine {
    /// Return the enumerated redexes (sorted by key) and the chosen indices.
    fn find_and_select(&mut self, st: &State, cfg: &Config, step: u64) -> (Vec<Redex>, Vec<usize>);
}

pub struct Host;

impl StepEngine for Host {
    fn find_and_select(&mut self, st: &State, cfg: &Config, step: u64) -> (Vec<Redex>, Vec<usize>) {
        let rs = find(st, cfg.finder);
        let chosen = choose(st, &rs, cfg, step);
        (rs, chosen)
    }
}

/// Apply the resolver to an enumerated, key-sorted redex set.
pub fn choose(st: &State, rs: &[Redex], cfg: &Config, step: u64) -> Vec<usize> {
    if rs.is_empty() {
        return Vec::new();
    }
    match cfg.resolver {
        Resolver::MaximalProgress => select_greedy(st, rs, &priority_order(rs, cfg.seed, step)),
        Resolver::SingleByPriority => vec![priority_order(rs, cfg.seed, step)[0]],
        Resolver::SingleUniform => {
            let h = mix64(mix64(mix64(cfg.seed) ^ step) ^ 0x5eed);
            vec![(h % rs.len() as u64) as usize]
        }
    }
}

/// Run to quiescence or a limit.
pub fn run(term: &Term, cfg: &Config, engine: &mut dyn StepEngine) -> RunReport {
    let mut it = InternTable::default();
    let mut st = match State::import(term, &mut it) {
        Ok(s) => s,
        Err(e) => {
            return RunReport { steps: vec![], stop: Stop::Budget(e.to_string()), fired: 0, final_term: term.clone() }
        }
    };
    run_state(&mut st, &mut it, cfg, engine)
}

pub fn run_state(st: &mut State, it: &mut InternTable, cfg: &Config, engine: &mut dyn StepEngine) -> RunReport {
    let mut steps = Vec::new();
    let mut fired = 0u64;
    let mut stop = Stop::MaxSteps;
    for t in 0..cfg.max_steps {
        let (rs, chosen) = engine.find_and_select(st, cfg, t);
        if chosen.is_empty() {
            stop = Stop::Quiescent;
            break;
        }
        let mut per_rule = [0u64; RULE_COUNT];
        for i in &chosen {
            per_rule[rs[*i].rule as usize] += 1;
        }
        let mut names_built = 0;
        if let Err(e) = commit_counting(st, it, &rs, &chosen, &mut names_built) {
            stop = Stop::Budget(format!("step {t}: {e}"));
            break;
        }
        fired += chosen.len() as u64;
        let state_hash = if cfg.hash_states { Some(st.export(it).content_hash()) } else { None };
        steps.push(StepRecord { step: t, enumerated: rs.len() as u64, fired: chosen.len() as u64, per_rule, names_built, state_hash });
        if st.rows() > cfg.max_rows {
            stop = Stop::Budget(format!("step {t}: more than {} live rows [comb-name-budget]", cfg.max_rows));
            break;
        }
    }
    RunReport { steps, stop, fired, final_term: st.export(it) }
}

#[cfg(test)]
mod tests;
