//! The kernels, generated: the rule table enters as constant arrays, so the
//! device dispatches over the same data the host does (Mat Decision 3.4).

use f1r3comb_term::rules::{rules, Datum, RULE_COUNT};
use f1r3comb_term::{Shape, SHAPES};

/// Layout of the `meta` buffer (u32 words), derived from the shape table and
/// the generated rule table.
pub mod meta {
    use super::{RULE_COUNT, SHAPES};
    pub const LEN: usize = 0; // SHAPES table lengths
    pub const BASE: usize = LEN + SHAPES; // SHAPES column bases into `cols`
    pub const SEG: usize = BASE + SHAPES; // rule segment bases + total
    pub const REF: usize = SEG + RULE_COUNT + 1; // row-ref bases + total
    pub const KEYS: usize = REF + SHAPES + 1; // bucket key space per datum table
    pub const SEED: usize = KEYS + 1; // seed hi, seed lo, step hi, step lo
    pub const W_COUNTS: usize = SEED + 4;
    pub const W_START: usize = W_COUNTS + 1;
    pub const W_BROWS: usize = W_START + 1;
    pub const W_CNT: usize = W_BROWS + 1;
    pub const W_OFF: usize = W_CNT + 1;
    pub const W_BEST: usize = W_OFF + 1;
    pub const W_TAKEN: usize = W_BEST + 1;
    pub const W_CTRL: usize = W_TAKEN + 1;
    pub const WORDS: usize = W_CTRL + 1;
}

/// Words per redex record in `rdx`: rule, consumer, p0, p1, p2, prio hi,
/// prio lo, decision (0 undecided, 1 taken, 2 excluded).
pub const REDEX_WORDS: usize = 8;

/// Uniform parameter slots (dynamic offsets, 256 bytes apart).
pub mod slot {
    pub const ZERO_COUNTS: u32 = 0;
    pub const SCAN_BUCKETS: u32 = 1;
    pub const SCAN_COUNTS: u32 = 2;
    pub const FILL_BEST: u32 = 3;
    pub const ZERO_TAKEN: u32 = 4;
    pub const ZERO_CTRL: u32 = 5;
    pub const REDEXES: u32 = 6;
    pub const COUNT: u32 = 7;
}

pub const WORKGROUP: u32 = 64;
pub const SCAN_WORKGROUP: u32 = 256;

/// Entry points, in the order a step dispatches them.
pub const ENTRY_POINTS: &[&str] = &[
    "k_fill", "k_hist", "k_scan", "k_scatter", "k_bsort", "k_count", "k_emit", "k_min_hi", "k_min_lo",
    "k_min_idx", "k_take", "k_exclude",
];

fn arr(name: &str, v: &[u32]) -> String {
    let body: Vec<String> = v.iter().map(|x| format!("{x}u")).collect();
    format!("var<private> {name}: array<u32, {}> = array<u32, {}>({});\n", v.len(), v.len(), body.join(", "))
}

/// The WGSL source of every kernel.
pub fn kernels() -> String {
    let mut cons = Vec::new();
    let mut nprem = Vec::new();
    let mut subj = Vec::new();
    let mut quiet = Vec::new();
    for r in rules() {
        cons.push(r.consumer.ix() as u32);
        nprem.push(r.premises.len() as u32);
        for j in 0..3 {
            match r.premises.get(j) {
                Some(p) => {
                    subj.push(p.subject as u32);
                    quiet.push((p.datum == Datum::Quiet) as u32);
                }
                None => {
                    subj.push(0);
                    quiet.push(0);
                }
            }
        }
    }
    let mut s = String::new();
    s.push_str("// F1R3Comb GPU kernels (generated from f1r3comb-term::rules; do not edit)\n");
    s.push_str(&format!(
        "// {} rules, {} shapes; see manifest.json for the buffer layout.\n\n",
        rules().len(),
        SHAPES
    ));
    s.push_str(&format!(
        "const M_IX: u32 = {}u;\nconst Q_IX: u32 = {}u;\nconst NSHAPES: u32 = {}u;\n",
        Shape::M.ix(),
        Shape::Q.ix(),
        SHAPES
    ));
    s.push_str(&arr("RULE_CONS", &cons));
    s.push_str(&arr("RULE_NPREM", &nprem));
    s.push_str(&arr("RULE_SUBJ", &subj));
    s.push_str(&arr("RULE_QUIET", &quiet));
    for (name, v) in [
        ("M_LEN", meta::LEN),
        ("M_BASE", meta::BASE),
        ("M_SEG", meta::SEG),
        ("M_REF", meta::REF),
        ("M_KEYS", meta::KEYS),
        ("M_SEED", meta::SEED),
        ("W_COUNTS", meta::W_COUNTS),
        ("W_START", meta::W_START),
        ("W_BROWS", meta::W_BROWS),
        ("W_CNT", meta::W_CNT),
        ("W_OFF", meta::W_OFF),
        ("W_BEST", meta::W_BEST),
        ("W_TAKEN", meta::W_TAKEN),
        ("W_CTRL", meta::W_CTRL),
    ] {
        s.push_str(&format!("const {name}: u32 = {v}u;\n"));
    }
    s.push_str(&format!("const NRULES: u32 = {}u;\nconst RW: u32 = {}u;\n", rules().len(), REDEX_WORDS));
    s.push_str(BODY);
    s
}

const BODY: &str = r#"
struct Params { a: u32, b: u32, c: u32, d: u32 }

@group(0) @binding(0) var<storage, read> mt: array<u32>;
@group(0) @binding(1) var<storage, read> cols: array<u32>;
@group(0) @binding(2) var<storage, read_write> work: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read_write> rdx: array<u32>;
@group(0) @binding(4) var<uniform> prm: Params;

fn col(s: u32, j: u32, i: u32) -> u32 {
    return cols[mt[M_BASE + s] + j * mt[M_LEN + s] + i];
}
fn tlen(s: u32) -> u32 { return mt[M_LEN + s]; }
fn wget(k: u32, i: u32) -> u32 { return atomicLoad(&work[mt[k] + i]); }
fn wset(k: u32, i: u32, v: u32) { atomicStore(&work[mt[k] + i], v); }

fn flat(g: vec3<u32>, n: vec3<u32>) -> u32 { return g.x + g.y * n.x * 64u; }

// ---- 64-bit arithmetic on (hi, lo) pairs: SplitMix64's finaliser
fn mul32(a: u32, b: u32) -> vec2<u32> {
    let a0 = a & 0xffffu; let a1 = a >> 16u;
    let b0 = b & 0xffffu; let b1 = b >> 16u;
    let p00 = a0 * b0; let p01 = a0 * b1; let p10 = a1 * b0; let p11 = a1 * b1;
    let mid = (p00 >> 16u) + (p01 & 0xffffu) + (p10 & 0xffffu);
    let lo = (p00 & 0xffffu) | (mid << 16u);
    let hi = p11 + (p01 >> 16u) + (p10 >> 16u) + (mid >> 16u);
    return vec2<u32>(hi, lo);
}
fn mul64(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> {
    let l = mul32(a.y, b.y);
    return vec2<u32>(l.x + a.x * b.y + a.y * b.x, l.y);
}
fn shr(a: vec2<u32>, s: u32) -> vec2<u32> {
    return vec2<u32>(a.x >> s, (a.y >> s) | (a.x << (32u - s)));
}
fn mix64(z0: vec2<u32>) -> vec2<u32> {
    var z = z0;
    z = mul64(z ^ shr(z, 30u), vec2<u32>(0xbf58476du, 0x1ce4e5b9u));
    z = mul64(z ^ shr(z, 27u), vec2<u32>(0x94d049bbu, 0x133111ebu));
    return z ^ shr(z, 31u);
}

// ---- utilities
@compute @workgroup_size(64)
fn k_fill(@builtin(global_invocation_id) g: vec3<u32>, @builtin(num_workgroups) n: vec3<u32>) {
    let i = flat(g, n);
    if (i < prm.b) { atomicStore(&work[prm.a + i], prm.c); }
}

var<workgroup> part: array<u32, 256>;

// Exclusive scan of work[a .. a+c) into work[b .. b+c], total at work[b+c].
@compute @workgroup_size(256)
fn k_scan(@builtin(local_invocation_id) l: vec3<u32>) {
    let t = l.x;
    let len = prm.c;
    let chunk = (len + 255u) / 256u;
    let lo = min(t * chunk, len);
    let hi = min(lo + chunk, len);
    var sum = 0u;
    for (var i = lo; i < hi; i = i + 1u) { sum = sum + atomicLoad(&work[prm.a + i]); }
    part[t] = sum;
    workgroupBarrier();
    for (var off = 1u; off < 256u; off = off * 2u) {
        var v = 0u;
        if (t >= off) { v = part[t - off]; }
        workgroupBarrier();
        part[t] = part[t] + v;
        workgroupBarrier();
    }
    var run = part[t] - sum;
    for (var i = lo; i < hi; i = i + 1u) {
        let x = atomicLoad(&work[prm.a + i]);
        atomicStore(&work[prm.b + i], run);
        run = run + x;
    }
    if (t == 255u) { atomicStore(&work[prm.b + len], part[255]); }
}

// ---- bucketing of the m and q subject columns (Mat Req. 5.3)
fn datum_key(i: u32) -> u32 {
    let ml = tlen(M_IX);
    if (i < ml) { return col(M_IX, 0u, i); }
    return mt[M_KEYS] + col(Q_IX, 0u, i - ml);
}

@compute @workgroup_size(64)
fn k_hist(@builtin(global_invocation_id) g: vec3<u32>, @builtin(num_workgroups) n: vec3<u32>) {
    let i = flat(g, n);
    if (i >= tlen(M_IX) + tlen(Q_IX)) { return; }
    atomicAdd(&work[mt[W_COUNTS] + datum_key(i)], 1u);
}

@compute @workgroup_size(64)
fn k_scatter(@builtin(global_invocation_id) g: vec3<u32>, @builtin(num_workgroups) n: vec3<u32>) {
    let i = flat(g, n);
    let ml = tlen(M_IX);
    if (i >= ml + tlen(Q_IX)) { return; }
    let k = datum_key(i);
    let pos = wget(W_START, k) + atomicSub(&work[mt[W_COUNTS] + k], 1u) - 1u;
    var row = i;
    if (i >= ml) { row = i - ml; }
    wset(W_BROWS, pos, row);
}

// Rows within a bucket ascending: the scatter's order is not deterministic.
@compute @workgroup_size(64)
fn k_bsort(@builtin(global_invocation_id) g: vec3<u32>, @builtin(num_workgroups) n: vec3<u32>) {
    let k = flat(g, n);
    if (k >= 2u * mt[M_KEYS]) { return; }
    let lo = wget(W_START, k);
    let hi = wget(W_START, k + 1u);
    for (var i = lo + 1u; i < hi; i = i + 1u) {
        let x = wget(W_BROWS, i);
        var j = i;
        loop {
            if (j <= lo) { break; }
            let y = wget(W_BROWS, j - 1u);
            if (y <= x) { break; }
            wset(W_BROWS, j, y);
            j = j - 1u;
        }
        wset(W_BROWS, j, x);
    }
}

// ---- redex enumeration in canonical order
fn rule_of(t: u32) -> u32 {
    var r = 0u;
    loop {
        if (r + 1u >= NRULES || t < mt[M_SEG + r + 1u]) { break; }
        r = r + 1u;
    }
    return r;
}

fn dshape(q: u32) -> u32 { if (q == 1u) { return Q_IX; } return M_IX; }

fn same_atom(s: u32, a: u32, b: u32) -> bool {
    return col(s, 0u, a) == col(s, 0u, b) && col(s, 1u, a) == col(s, 1u, b);
}

fn prio(r: u32, c: u32, np: u32, p: array<u32, 3>) -> vec2<u32> {
    var h = mix64(vec2<u32>(mt[M_SEED], mt[M_SEED + 1u]));
    h = mix64(h ^ vec2<u32>(mt[M_SEED + 2u], mt[M_SEED + 3u]));
    h = mix64(h ^ vec2<u32>(0u, r));
    h = mix64(h ^ vec2<u32>(0u, c));
    for (var j = 0u; j < np; j = j + 1u) { h = mix64(h ^ vec2<u32>(0u, p[j])); }
    return h;
}

fn enumerate(t: u32, emit: bool) -> u32 {
    let r = rule_of(t);
    let c = t - mt[M_SEG + r];
    let cs = RULE_CONS[r];
    let np = RULE_NPREM[r];
    let nk = mt[M_KEYS];
    var lo: array<u32, 3>;
    var hi: array<u32, 3>;
    var ix: array<u32, 3>;
    var ds: array<u32, 3>;
    for (var j = 0u; j < 3u; j = j + 1u) {
        if (j < np) {
            let name = col(cs, RULE_SUBJ[3u * r + j], c);
            let q = RULE_QUIET[3u * r + j];
            ds[j] = q;
            if (name >= nk) { return 0u; }
            let key = name + q * nk;
            lo[j] = wget(W_START, key);
            hi[j] = wget(W_START, key + 1u);
            if (lo[j] == hi[j]) { return 0u; }
        } else {
            lo[j] = 0u; hi[j] = 1u; ds[j] = 0u;
        }
        ix[j] = lo[j];
    }
    var count = 0u;
    var base = 0u;
    if (emit) { base = wget(W_OFF, t); }
    loop {
        var p: array<u32, 3>;
        for (var j = 0u; j < 3u; j = j + 1u) {
            if (j < np) { p[j] = wget(W_BROWS, ix[j]); } else { p[j] = 0xffffffffu; }
        }
        var ok = true;
        for (var a = 0u; a < np; a = a + 1u) {
            for (var b = a + 1u; b < np; b = b + 1u) {
                if (ds[a] == ds[b]) {
                    if (p[a] == p[b]) { ok = false; }
                    if (p[b] < p[a] && same_atom(dshape(ds[a]), p[a], p[b])) { ok = false; }
                }
            }
        }
        if (ok) {
            if (emit) {
                let o = (base + count) * RW;
                rdx[o] = r; rdx[o + 1u] = c;
                rdx[o + 2u] = p[0]; rdx[o + 3u] = p[1]; rdx[o + 4u] = p[2];
                let h = prio(r, c, np, p);
                rdx[o + 5u] = h.x; rdx[o + 6u] = h.y; rdx[o + 7u] = 0u;
            }
            count = count + 1u;
        }
        // odometer, last premise fastest
        var k = np;
        var done = true;
        loop {
            if (k == 0u) { break; }
            k = k - 1u;
            ix[k] = ix[k] + 1u;
            if (ix[k] < hi[k]) { done = false; break; }
            ix[k] = lo[k];
        }
        if (done) { break; }
    }
    return count;
}

@compute @workgroup_size(64)
fn k_count(@builtin(global_invocation_id) g: vec3<u32>, @builtin(num_workgroups) n: vec3<u32>) {
    let t = flat(g, n);
    if (t >= mt[M_SEG + NRULES]) { return; }
    wset(W_CNT, t, enumerate(t, false));
}

@compute @workgroup_size(64)
fn k_emit(@builtin(global_invocation_id) g: vec3<u32>, @builtin(num_workgroups) n: vec3<u32>) {
    let t = flat(g, n);
    if (t >= mt[M_SEG + NRULES]) { return; }
    enumerate(t, true);
}

// ---- the greedy maximal independent set, in rounds (Mat Prop. 6.4)
fn nrefs(i: u32) -> u32 { return 1u + RULE_NPREM[rdx[i * RW]]; }
fn ref_of(i: u32, k: u32) -> u32 {
    let r = rdx[i * RW];
    if (k == 0u) { return mt[M_REF + RULE_CONS[r]] + rdx[i * RW + 1u]; }
    return mt[M_REF + dshape(RULE_QUIET[3u * r + k - 1u])] + rdx[i * RW + 1u + k];
}
fn nref_total() -> u32 { return mt[M_REF + NSHAPES]; }
fn best(part: u32, rf: u32) -> u32 { return mt[W_BEST] + part * nref_total() + rf; }

@compute @workgroup_size(64)
fn k_min_hi(@builtin(global_invocation_id) g: vec3<u32>, @builtin(num_workgroups) n: vec3<u32>) {
    let i = flat(g, n);
    if (i >= prm.a || rdx[i * RW + 7u] != 0u) { return; }
    for (var k = 0u; k < nrefs(i); k = k + 1u) { atomicMin(&work[best(0u, ref_of(i, k))], rdx[i * RW + 5u]); }
}

@compute @workgroup_size(64)
fn k_min_lo(@builtin(global_invocation_id) g: vec3<u32>, @builtin(num_workgroups) n: vec3<u32>) {
    let i = flat(g, n);
    if (i >= prm.a || rdx[i * RW + 7u] != 0u) { return; }
    for (var k = 0u; k < nrefs(i); k = k + 1u) {
        let rf = ref_of(i, k);
        if (atomicLoad(&work[best(0u, rf)]) == rdx[i * RW + 5u]) {
            atomicMin(&work[best(1u, rf)], rdx[i * RW + 6u]);
        }
    }
}

@compute @workgroup_size(64)
fn k_min_idx(@builtin(global_invocation_id) g: vec3<u32>, @builtin(num_workgroups) n: vec3<u32>) {
    let i = flat(g, n);
    if (i >= prm.a || rdx[i * RW + 7u] != 0u) { return; }
    for (var k = 0u; k < nrefs(i); k = k + 1u) {
        let rf = ref_of(i, k);
        if (atomicLoad(&work[best(0u, rf)]) == rdx[i * RW + 5u] && atomicLoad(&work[best(1u, rf)]) == rdx[i * RW + 6u]) {
            atomicMin(&work[best(2u, rf)], i);
        }
    }
}

@compute @workgroup_size(64)
fn k_take(@builtin(global_invocation_id) g: vec3<u32>, @builtin(num_workgroups) n: vec3<u32>) {
    let i = flat(g, n);
    if (i >= prm.a || rdx[i * RW + 7u] != 0u) { return; }
    for (var k = 0u; k < nrefs(i); k = k + 1u) {
        if (atomicLoad(&work[best(2u, ref_of(i, k))]) != i) { return; }
    }
    rdx[i * RW + 7u] = 1u;
    for (var k = 0u; k < nrefs(i); k = k + 1u) { wset(W_TAKEN, ref_of(i, k), 1u); }
}

@compute @workgroup_size(64)
fn k_exclude(@builtin(global_invocation_id) g: vec3<u32>, @builtin(num_workgroups) n: vec3<u32>) {
    let i = flat(g, n);
    if (i >= prm.a || rdx[i * RW + 7u] != 0u) { return; }
    for (var k = 0u; k < nrefs(i); k = k + 1u) {
        if (wget(W_TAKEN, ref_of(i, k)) != 0u) { rdx[i * RW + 7u] = 2u; return; }
    }
    atomicAdd(&work[mt[W_CTRL]], 1u);
}
"#;
