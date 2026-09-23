//! Rebuilding a normal form: substitution, shifting and free-level remapping
//! (spec §7, Decision 1.7). One iterative walk serves all three.
//!
//! Every rebuild re-runs the smart constructors, so the result is in normal
//! form: parallel components re-ordered, binds re-ordered, encodings recomputed.
//! Re-ordering the binds of a receipt permutes its binder slots, so the body's
//! indices are permuted with them; that is the correctness half of the
//! incremental-renormalisation obligation, whose performance half (recompute
//! only the ancestors of substitution sites) is left for v1.1.

use crate::term::*;

/// A value substituted for a binder slot.
#[derive(Clone, Debug)]
pub enum Val {
    Name(Name),
    Proc(Norm),
}

/// Substitution for the innermost binder group: `vals[j]` replaces de Bruijn
/// index `j`.
#[derive(Clone, Debug, Default)]
pub struct Subst {
    pub vals: Vec<Val>,
}

impl Subst {
    /// Build a substitution from bindings given in *slot* order (slot 0 first),
    /// which is the order the store returns them in. Slot `s` of a group of `k`
    /// is de Bruijn index `k - 1 - s`.
    pub fn from_slots(mut slots: Vec<Val>) -> Subst {
        slots.reverse();
        Subst { vals: slots }
    }
    pub fn len(&self) -> usize {
        self.vals.len()
    }
    pub fn is_empty(&self) -> bool {
        self.vals.is_empty()
    }
}

enum Xf<'a> {
    Subst(&'a Subst),
    Shift { by: i64, cutoff: u32 },
    Remap(&'a [u32]),
    Ground(&'a [[u8; 32]]),
    SlotPerm { map: &'a [u32], k: u32 },
}

enum R {
    P(Norm, u32),
    N(Name, u32),
    MkPar(u32),
    MkSendHead(bool, u32),
    MkReceive(Vec<NBind>, u32),
    MkNew(u32),
    MkEval,
    MkQuote,
}

impl Norm {
    /// Substitute the innermost binder group (spec §7 API).
    pub fn substitute(&self, env: &Subst) -> Norm {
        if env.is_empty() {
            return self.clone();
        }
        rebuild(self, &Xf::Subst(env))
    }
    /// Shift bound indices at or above `cutoff` by `by`.
    pub fn shift(&self, by: i64, cutoff: u32) -> Norm {
        if by == 0 {
            return self.clone();
        }
        rebuild(self, &Xf::Shift { by, cutoff })
    }
    /// Relabel free levels: `map[old] = new`.
    pub fn remap_free(&self, map: &[u32]) -> Norm {
        rebuild(self, &Xf::Remap(map))
    }
    /// Ground free names: level `l` becomes the unforgeable name `keys[l]`.
    /// This is how an executive closes a term before running it, and it is what
    /// makes two deploys that both write `stdout` meet on one channel.
    pub fn ground_free(&self, keys: &[[u8; 32]]) -> Norm {
        rebuild(self, &Xf::Ground(keys))
    }
    /// Ground free names by identifier, with a stable derivation.
    pub fn ground_free_by(&self, idents: &[String]) -> Norm {
        let keys: Vec<[u8; 32]> = idents.iter().map(|i| free_key(i)).collect();
        self.ground_free(&keys)
    }
    fn perm_slots(&self, map: &[u32], k: u32) -> Norm {
        rebuild(self, &Xf::SlotPerm { map, k })
    }
}

fn shift_val(v: &Val, d: u32) -> Val {
    if d == 0 {
        return v.clone();
    }
    match v {
        Val::Proc(p) => Val::Proc(p.shift(d as i64, 0)),
        Val::Name(Name::Quote(q)) => Val::Name(Norm::quote(q.shift(d as i64, 0))),
        Val::Name(Name::Bound(i)) => Val::Name(Name::Bound(i + d)),
        Val::Name(other) => Val::Name(other.clone()),
    }
}

fn apply_index(xf: &Xf<'_>, i: u32, d: u32) -> Option<u32> {
    match xf {
        Xf::Shift { by, cutoff } => {
            if i >= *cutoff + d {
                Some(((i as i64) + by).max(0) as u32)
            } else {
                Some(i)
            }
        }
        Xf::SlotPerm { map, k } => {
            if i >= d && i - d < *k {
                let j = i - d;
                let s_old = *k - 1 - j;
                let s_new = map[s_old as usize];
                Some(d + (*k - 1 - s_new))
            } else {
                Some(i)
            }
        }
        _ => Some(i),
    }
}

fn rebuild(t: &Norm, xf: &Xf<'_>) -> Norm {
    let mut steps: Vec<R> = vec![R::P(t.clone(), 0)];
    let mut procs: Vec<Norm> = Vec::new();
    let mut names: Vec<Name> = Vec::new();
    while let Some(step) = steps.pop() {
        match step {
            R::P(p, d) => match p.node() {
                Node::Nil | Node::Wild => procs.push(p.clone()),
                Node::FreeVar(l) => procs.push(match xf {
                    Xf::Remap(map) => Norm::free_var(map.get(*l as usize).copied().unwrap_or(*l)),
                    Xf::Ground(keys) => match keys.get(*l as usize) {
                        Some(k) => Norm::eval(Name::Unforgeable(*k)),
                        None => p.clone(),
                    },
                    _ => p.clone(),
                }),
                Node::BoundVar(i) => {
                    let i = *i;
                    match xf {
                        Xf::Subst(s) => {
                            if i >= d && ((i - d) as usize) < s.vals.len() {
                                match shift_val(&s.vals[(i - d) as usize], d) {
                                    Val::Proc(q) => procs.push(q),
                                    Val::Name(n) => procs.push(Norm::eval(n)),
                                }
                            } else if i >= d {
                                procs.push(Norm::bound_var(i - s.vals.len() as u32));
                            } else {
                                procs.push(p.clone());
                            }
                        }
                        _ => procs.push(Norm::bound_var(apply_index(xf, i, d).unwrap_or(i))),
                    }
                }
                Node::Par(v) => {
                    steps.push(R::MkPar(v.len() as u32));
                    for c in v {
                        steps.push(R::P(c.clone(), d));
                    }
                }
                Node::Send {
                    chan,
                    persistent,
                    args,
                } => {
                    steps.push(R::MkSendHead(*persistent, args.len() as u32));
                    for a in args {
                        steps.push(R::P(a.clone(), d));
                    }
                    steps.push(R::N(chan.clone(), d));
                }
                Node::New { count, body } => {
                    steps.push(R::MkNew(*count));
                    steps.push(R::P(body.clone(), d + count));
                }
                Node::Eval(n) => {
                    steps.push(R::MkEval);
                    steps.push(R::N(n.clone(), d));
                }
                Node::Receive { binds, body } => {
                    let k: u32 = binds.iter().map(|b| b.binders as u32).sum();
                    steps.push(R::MkReceive(binds.clone(), d));
                    steps.push(R::P(body.clone(), d + k));
                    // Channels are in the enclosing scope and are rebuilt;
                    // patterns are closed with respect to it and are copied.
                    for b in binds.iter().rev() {
                        steps.push(R::N(b.chan.clone(), d));
                    }
                }
            },
            R::N(n, d) => match &n {
                Name::Quote(q) => {
                    steps.push(R::MkQuote);
                    steps.push(R::P(q.clone(), d));
                }
                Name::Unforgeable(_) => names.push(n.clone()),
                Name::Free(l) => names.push(match xf {
                    Xf::Remap(map) => Name::Free(map.get(*l as usize).copied().unwrap_or(*l)),
                    Xf::Ground(keys) => match keys.get(*l as usize) {
                        Some(k) => Name::Unforgeable(*k),
                        None => n.clone(),
                    },
                    _ => n.clone(),
                }),
                Name::Bound(i) => {
                    let i = *i;
                    match xf {
                        Xf::Subst(s) => {
                            if i >= d && ((i - d) as usize) < s.vals.len() {
                                match shift_val(&s.vals[(i - d) as usize], d) {
                                    Val::Name(m) => names.push(m),
                                    Val::Proc(q) => names.push(Norm::quote(q)),
                                }
                            } else if i >= d {
                                names.push(Name::Bound(i - s.vals.len() as u32));
                            } else {
                                names.push(Name::Bound(i));
                            }
                        }
                        _ => names.push(Name::Bound(apply_index(xf, i, d).unwrap_or(i))),
                    }
                }
            },
            R::MkQuote => {
                let p = procs.pop().expect("quote operand");
                names.push(Norm::quote(p));
            }
            R::MkEval => {
                let n = names.pop().expect("eval operand");
                procs.push(Norm::eval(n));
            }
            R::MkPar(n) => {
                let at = procs.len() - n as usize;
                let parts = procs.split_off(at);
                procs.push(Norm::par(parts));
            }
            R::MkSendHead(persistent, n) => {
                let at = procs.len() - n as usize;
                let args = procs.split_off(at);
                let chan = names.pop().expect("send channel");
                procs.push(Norm::send(chan, persistent, args));
            }
            R::MkNew(k) => {
                let body = procs.pop().expect("new body");
                procs.push(Norm::new_scope(k, body));
            }
            R::MkReceive(old, _d) => {
                let body = procs.pop().expect("receive body");
                let at = names.len() - old.len();
                let chans = names.split_off(at);
                let binds: Vec<NBind> = old
                    .iter()
                    .zip(chans)
                    .map(|(b, c)| NBind::new(b.kind, c, b.pats.clone(), b.binders))
                    .collect();
                procs.push(receive_permuting(binds, body));
            }
        }
    }
    procs.pop().expect("rebuilt term")
}

/// Build a receive from binds in an arbitrary order, permuting the body's
/// binder slots to follow the order the encoding will impose.
pub fn receive_permuting(binds: Vec<NBind>, body: Norm) -> Norm {
    let mut binds = binds;
    let old_bases: Vec<u32> = {
        let mut acc = 0u32;
        binds
            .iter()
            .map(|b| {
                let x = acc;
                acc += b.binders as u32;
                x
            })
            .collect()
    };
    let k: u32 = binds.iter().map(|b| b.binders as u32).sum();
    let perm = sort_binds(&mut binds); // perm[new_pos] = old_pos
    let identity = perm.iter().enumerate().all(|(i, j)| i == *j);
    let body = if identity || k == 0 {
        body
    } else {
        let mut map = vec![0u32; k as usize];
        let mut base = 0u32;
        for (_new_pos, old_pos) in perm.iter().enumerate() {
            let b = &binds[_new_pos];
            for s in 0..b.binders as u32 {
                map[(old_bases[*old_pos] + s) as usize] = base + s;
            }
            base += b.binders as u32;
        }
        body.perm_slots(&map, k)
    };
    Norm::receive(binds, body)
}

/// The channel a free identifier grounds to. Stable across runs and targets.
pub fn free_key(ident: &str) -> [u8; 32] {
    let mut pre = Vec::with_capacity(16 + ident.len());
    pre.extend_from_slice(b"k1ndl1ng/free/");
    pre.extend_from_slice(ident.as_bytes());
    crate::hash::blake2b_256(&pre).0
}
