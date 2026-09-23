//! Addresses, leaves and static channels (draft 3 §6.3, §6.5; F1R3Comb v0.6
//! §5.3, §5.6, §9.2).
//!
//! * `K = @m(@0,@0)`, `L σ = @m(σ,@0)`, `R σ = @m(σ,K)` (Def. 6.6); `σ·w` is
//!   the leaf at path `w`. `L` and `R` are injective with disjoint images and
//!   strictly increase size (Lem. 6.7, 6.14).
//! * Inside a compiled unit the address is not known: leaves are written
//!   beneath a hole, `□·w`, and `inst` fills the hole with the address.
//! * The static channels `R_STAR` (where every unit instance receives its
//!   address) and `F_STAR` (where `inst` delivers the instantiated unit) are
//!   shared by every instance of every unit. See the crate README for why
//!   they are global rather than per unit.
//! * Root addresses come from [`Allocator`]: each exceeds every name of the
//!   program in size (Req. 5.28) and roots are pairwise prefix-incomparable
//!   (Req. 5.32).

use crate::{Atom, CName, Shape, Term};

/// `K = @m(@0, @0)`.
pub fn k_const() -> CName {
    CName::quote(Term::atom(Atom::m(CName::nil(), CName::nil())))
}

/// `L σ` (`false`) or `R σ` (`true`).
pub fn step(sigma: &CName, bit: bool) -> CName {
    let second = if bit { k_const() } else { CName::nil() };
    CName::quote(Term::atom(Atom::m(sigma.clone(), second)))
}

/// `σ·w`.
pub fn leaf(sigma: &CName, path: &[bool]) -> CName {
    let mut cur = sigma.clone();
    for b in path {
        cur = step(&cur, *b);
    }
    cur
}

/// If `n` is `σ·w` for some `w` (possibly empty) with `σ` not itself of
/// the form `L τ` / `R τ`, return `(σ, w)`.
pub fn spine(n: &CName) -> (CName, Vec<bool>) {
    let k = k_const();
    let mut bits = Vec::new();
    let mut cur = n.clone();
    loop {
        let t = cur.drop();
        match t.atoms() {
            [a] if a.shape() == Shape::M && (a.names()[1] == CName::nil() || a.names()[1] == k) && cur.proc_ref().is_some() => {
                bits.push(a.names()[1] == k);
                cur = a.names()[0].clone();
            }
            _ => break,
        }
    }
    bits.reverse();
    (cur, bits)
}

/// Is `n` a leaf (at any depth, including the root itself) beneath a root
/// address — i.e. does its `L/R` spine bottom out in a root?
pub fn is_address(n: &CName) -> bool {
    is_root(&spine(n).0)
}

/// The shape of a root: `@bl(@k^i(@0), M)`, a `bl` node no translated name
/// has at top level.
pub fn is_root(n: &CName) -> bool {
    matches!(n.drop().atoms(), [a] if a.shape() == Shape::Bl)
}

/// `σ ≤ τ`: `τ` is a leaf beneath `σ` (or equal).
pub fn is_prefix(sigma: &CName, tau: &CName) -> bool {
    let mut cur = tau.clone();
    loop {
        if cur == *sigma {
            return true;
        }
        let t = cur.drop();
        match t.atoms() {
            [a] if a.shape() == Shape::M => cur = a.names()[0].clone(),
            _ => return false,
        }
    }
}

/// Where every unit instance receives its address: `@k(@0)`.
pub fn r_star() -> CName {
    CName::quote(Term::atom(Atom::k(CName::nil())))
}

/// Where every `inst` delivers its instantiated unit: `@k(@k(@0))`.
pub fn f_star() -> CName {
    CName::quote(Term::atom(Atom::k(r_star())))
}

/// The leaf at `w` beneath the hole of context level `k`.
pub fn hole_leaf(k: u32, path: &[bool]) -> CName {
    leaf(&CName::hole(k), path)
}

/// Elias-gamma code of `j >= 1`: a prefix-free set of paths, so the
/// subtrees beneath distinct spare leaves are disjoint (`[nowns]`).
pub fn gamma(j: u32) -> Vec<bool> {
    assert!(j >= 1);
    let n = 31 - j.leading_zeros();
    let mut v = vec![false; n as usize];
    for i in (0..=n).rev() {
        v.push((j >> i) & 1 == 1);
    }
    v
}

/// The root-address allocator of Req. 9.6: roots are `@bl(@k^i(@0), M)` for
/// `i = 1, 2, …`, where `M` is the largest name of the program (so every root
/// is larger, Req. 5.28), and no root is a leaf beneath another (a leaf is
/// `m`-shaped; a root is `bl`-shaped), so roots are prefix-incomparable
/// (Req. 5.32).
#[derive(Clone, Debug, Default)]
pub struct Allocator {
    issued: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AddrError {
    /// `comb-address-too-small`.
    TooSmall,
    /// `comb-address-overlap`.
    Overlap,
}

impl Allocator {
    pub fn new() -> Allocator {
        Allocator::default()
    }

    /// A fresh root larger than `bound`.
    pub fn root(&mut self, bound: &CName) -> CName {
        self.issued += 1;
        let mut c = CName::nil();
        for _ in 0..self.issued {
            c = CName::quote(Term::atom(Atom::k(c)));
        }
        CName::quote(Term::atom(Atom::bl(c, bound.clone())))
    }

    /// Deploy checks on a proposed root against the program bound and the
    /// roots in use.
    pub fn check(root: &CName, bound: &CName, in_use: &[CName]) -> Result<(), AddrError> {
        if root.size() <= bound.size() {
            return Err(AddrError::TooSmall);
        }
        if in_use.iter().any(|u| is_prefix(u, root) || is_prefix(root, u)) {
            return Err(AddrError::Overlap);
        }
        Ok(())
    }
}

/// The largest closed name occurring in `t` (through stores and contexts):
/// the size bound roots must exceed.
pub fn largest_name(t: &Term) -> CName {
    let mut best = CName::nil();
    let mut work = vec![t.clone()];
    let mut seen = std::collections::HashSet::new();
    while let Some(t) = work.pop() {
        for a in t.atoms() {
            for n in a.names() {
                if n.is_closed() && n.size() > best.size() {
                    best = n.clone();
                }
                if let Some(p) = n.proc_ref() {
                    if seen.insert(n.content_hash()) {
                        work.push(p.clone());
                    }
                }
            }
            if let Some(p) = a.store().or(a.context()) {
                work.push(p.clone());
            }
        }
    }
    best
}
