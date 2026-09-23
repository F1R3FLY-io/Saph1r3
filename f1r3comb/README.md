# f1r3comb — a two-phase compiler from the rho calculus to the rho combinators, and to the GPU

```
K0 rho ─parse/normalise─▶ Norm ─phase 1─▶ RC^ν (binder) ─phase 2─▶ target (inst erection) ─deploy─▶ matrix machine / GPU
 (.rho)                                    (IR text)                (.comb, target text)             (.cmat, bundle)
```

This workspace implements `rho-combinators/` in F1R3FLY-io/publications at `3dbf4b9`:

* the two-phase translation of **draft 3 §6**. Phase one compiles into the combinators
  extended with `new`. Phase two eliminates the binder by running each instance at a
  run-time address and erecting it with the context-instantiation combinator `inst` (§6.7);
* the **F1R3Comb v0.6** compiler specification, presentation A, with `inst` as the erection
  and the curried erection as the `--scheme curried` build;
* the **F1R3Comb-Mat v0.5** bulk-synchronous matrix machine, host and wgpu device, with
  `inst` and the curried `cstar` sharing one instantiation kernel.

Each stage's artefact can be saved, and any artefact can be fed back in as input.

## Build

The front end is the K1ndl1ng parser and normaliser from campf1r3, checked out as a sibling:

```
git clone https://github.com/F1R3FLY-io/campf1r3        # tested at b46e16c25458df8d2309554b316f3818a8e0a276
cd f1r3comb
cargo build --release                                     # binary: target/release/f1r3comb
cargo test --workspace --features f1r3comb-check/gpu      # 48 tests; device tests skip without an adapter
```

The GPU backend is wgpu 25 (Vulkan, Metal, DX12). A software Vulkan driver is enough:
the tests here ran on Mesa lavapipe (`apt install mesa-vulkan-drivers`).

## Use

```
f1r3comb compile examples/w.rho -o w.comb --emit-ir-text w.txt --emit-gpu w.gpu/
f1r3comb run w.comb --trace                 # deploys at a fresh root address, runs on the host
f1r3comb run w.gpu --machine gpu            # the bundle is the deployed state
f1r3comb lattice examples/w.rho             # phase-one and phase-two points
f1r3comb compile examples/race.rho --scheme curried   # the logic's build, where it reaches
f1r3comb run w.comb --resolver cross-instance         # the adversarial scheduler (host only)
f1r3comb compile examples/w.rho --scheme positional -o wp.comb   # the unsound negative control
f1r3comb run examples/dx.rho --max-steps 200 --trace             # the recursion combinator
```

* **`.comb` v2.** The header records presentation A, shortcuts off, the erection scheme,
  the source hash, F(P) as used, and the marking analysis (receives, and how many of them
  are unsafe for the fixed-arity family). It also records the size bound a root address
  must exceed. The body is the canonical target encoding.
* **Deploying.** `run` posts a fresh root address at `r*`. A deploy the root allocator would
  reject is refused.
* **The bundle.** It holds `program.cmat`, the deployed state, with holes as entries of
  their own (`.cmat` v2). It also holds `kernels.wgsl`, generated from the rule table for
  21 shapes and 20 rules, and `manifest.json`. A curried artefact is rejected before it
  reaches a device (`comb-mat-scheme`).

## The translation

**Phase one** (`f1r3comb-lower::phase1`, output type `f1r3comb-ir::Node`) is Def. 6.3: a
homomorphism on the normal form. Each input becomes
`new p0..pk, c.., q.., b, c (D | G | links)`. No name is allocated. The IR has its own type,
with a `New` node, and a nameless canonical encoding (de Bruijn indices), so α-equivalence
is equality of encodings.

**Phase two** (`phase2`) compiles every *unit* whose own level binds names:

```
(atoms not mentioning its names) | inst(r*, f*, C) | e(f*)
```

Each bound name `z_j` is written in the context `C` as the leaf `□·γ(j)` beneath the hole.
Here `γ` is the Elias-gamma code, so the paths are prefix-free and spare addresses own
disjoint subtrees. A unit is the top level, the body a gate stores, or any quoted process.
An instance receives an address σ at `r*`. `inst` delivers `@C[σ]` at `f*`, and `e(f*)`
erects it: two steps, one premise, and one address input at a static channel.

**Nested units.** A stored body that has its own inputs is a unit inside its parent's
context. Holes carry a **de Bruijn index over context nodes**: `□0` is the innermost
context's hole, and `□1` refers to the parent's. `inst` fills its own hole and shifts
deeper ones, so filling a parent never touches a child's hole (`f1r3comb_term::fill`,
`InternTable::fill`).

**Addresses.** An address is supplied **per release site**. Phase one emits
`new a (m(r*, a))` for every drop `*y` (o4) and for every gate whose body contains an
input. After phase two, `a` is a leaf of the instance's address, and the unit released
there takes it. Every run-time instantiation therefore has an address posted by the event
that caused it. Root addresses come from `f1r3comb_term::addr::Allocator`. Each root has
the form `@bl(@k^i(@0), M)`, where `M` is the program's largest name. That makes the root
larger than every name of the image, and roots are pairwise prefix-incomparable.

**Post-passes** (`verify`) run in the default build and fail compilation:

* the single-address-input discipline, Def. 6.10 (`comb-discipline`);
* no address feeding an (o5) chain (`comb-o5-address`);
* no static channel occurring as a value (`comb-static-not-fresh`);
* no hole outside a context (`comb-context-misplaced`).

**The curried build** (`--scheme curried`, Req. 5.27) emits one `cstar(t, f, T) | e(f)`
per atom mentioning the unit's names. `T` is that atom with its names as paths beneath the
hole, and a `d`-tree on static channels fans the address out to each `cstar`. The gates'
`b` become static names (unmarked, Rem. 6.20), as in `phase2.py`, so a stored body with
one scoped atom is built by the recipe of Req. 4.9: `cstar` for the atom, `cons_|` with
the closed rest, then `cons_q`. A body with two or more scoped atoms needs a join of two
address-derived names at static channels. The compiler reports this as
`comb-curried-store`, and the discipline check rejects it. An atom that mentions an
enclosing unit's names has no curried template at all (`comb-curried-reach`). The static
channels are functions of the unit's IR hash, a role and an index (Req. 6.4).

**Marking** is draft 3 Rem. 6.20. Each marked name belongs to one (receive, occurrence)
flow. A receive is unsafe if some non-distributor atom joins two flows.

## Crates

| crate | contents |
|---|---|
| `f1r3comb-term` | Target: 21 shapes (9 atoms, `cons_|`, a constructor per atom shape, `inst`, `cstar`), the structured tag layout, holes and contexts, `fill`, encoding/hashing/decoding, the **generated** rule table, addresses and the allocator, the printer, lattice, `.comb` v2. |
| `f1r3comb-ir` | RC^ν: `New`, variables, nameless encoding, α-equivalence, free variables, counts, rendering. |
| `f1r3comb-lower` | Checks, phase one, phase two (`inst`; `curried`; `positional` as negative control), post-passes with address propagation, marking, header. |
| `f1r3comb-mat` | Interned state with a hole kind and a `free` column; templates as the third column; `fill` over the interned form, visiting only hole-bearing names; `.cmat` v2. |
| `f1r3comb-par` | Host machine over the generated table; `Instantiate` conclusions for `inst`/`cstar` (with the curried validity check); names built per step; `deploy`; the `cross-instance` adversarial resolver. |
| `f1r3comb-gpu` | WGSL generated for any shape and rule count; `inst` only on the device. |
| `f1r3comb-check` | K0 reference reducer, observation over encoded names, the corpus, W and its variants, the E5–E7 experiments, the compiled D_x, both schemes, golden files (`golden/`, regenerated only with `F1R3COMB_BLESS=1`). |
| `f1r3comb-cli` | The binary. |

## What is verified (48 tests)

| test | what it checks | result |
|---|---|---|
| W regression, 100 seeds, host (Obl. 11.1) | W and W3 agree with the source on every seed; the crossed residue `m(@X, Y)` never appears | pass |
| W on the device | step traces identical to the host | pass |
| Positional negative control | W crosses on 49/100 seeds and W3 on 85/100; the near miss `for(y<-a){*y\|*y}` never | as expected |
| Marking (Obl. 11.4) | R is unsafe and the near miss safe; this agrees with the positional oracle over 100 seeds at 2 and 3 instances | agrees |
| Corpus | 19 programs covering every occurrence kind, nesting, relays, (o5) chains, races, and received code with inputs run once and twice; 4 seeds on the host, 2 on the device | pass |
| Address independence (Obl. 11.14) | the corpus at two roots gives identical observations | pass |
| Conflation, freshness, discipline | `Q \| m(@Q,@P)` is stuck; the adversarial source name `@(x!(Nil))` is not apparatus and the root exceeds it; a literal erection is rejected with `comb-discipline` and `comb-o5-address` | pass |
| Canonicity | permuted pars are α-equal at phase one and byte-equal at phase two | pass |
| Golden (ii) | R's image binds its proxies, and W's image posts one address per drop | pass |
| Golden files (Req. 8.8) | W and D_x: source, IR, target, encoding, hash, full reduction trace; in the W file, the two R instances are erected in the same step and the observation is `@X!(X)`, `@Y!(Y)` | pass |
| Both schemes on the corpus (Obl. 11.5) | curried reaches 12 of 21 programs and agrees with `inst` and the source on 8 seeds each; 4 need a joined store; 5 mention an enclosing unit's names | pass |
| Contexts round-trip (Obl. 11.6) | on 6 units, `C[σ]` equals the curried erection run to completion, byte for byte, except the gates' `b` | pass |
| Adversarial scheduler (Obl. 11.2) | W, W3 and the near miss under `inst`, 100 seeds: 0 disagreements. E5 at 2 instances: literal mixes on 50/50 runs (23/50 under maximal progress); curried and `inst`, 2–8 instances: 0 | pass |
| Nesting family | phase-one atoms grow linearly in nesting depth | pass |

**E5–E7, three arms**, run on this machine with the images `phase2.py` builds:

| experiment | literal | curried | inst | `phase2-results.txt` |
|---|---|---|---|---|
| E5 parallel steps | 5 | 3 | 2 | 5 / 3 / 2 |
| E5 sequential steps, n = 2 | 16 | 10 | 4 | 16 / 10 / 4 |
| E5 runs with mixing, n = 2 (parallel) | 23 / 50 | 0 | 0 | 29 / 0 / 0 |
| E5 runs with mixing, n ≥ 4 | 48–50 / 50 | 0 | 0 | 48–50 / 0 / 0 |
| E6 D_x, parallel steps per unfolding | — | {10} | {7} | {10} / {7} |
| E6 D_x, sequential mean | — | 32.00 | 10.00 | 32.00 / 10.00 |
| E6 D_x, names per unfolding | — | {22} | {16} | {22} / {16} |
| E7 mixed stores (2 / 8 instances) | — | 60/100, 349/400 | 0, 0 | 44/100, 347/400 / 0, 0 |

The literal arm's mixing counts differ in detail from `phase2-results.txt` because the
random priorities differ. The E6 figures reproduce exactly.

**D_x as this compiler emits it** runs 200 unfoldings at {7} parallel steps and a sequential
mean of 10.00 per unfolding, the paper's figures, and builds {20} names per unfolding
against the hand image's 16. The four extra names come from phase one: the compiled unit
allocates a separate gate pair `b`, `c` and a posted address for the drop.

## Decisions and deviations

1. **Global static channels `r*` and `f*`** are shared by every unit, rather than one `r`
   per unit named by the unit's hash (Req. 6.4). A per-unit `r` cannot serve code that is
   *received* and then dropped, because the drop site cannot know which unit it will run.
   The spec provides only for nested units and self-instantiation. With fresh addresses
   posted per release site, a shared `r*` is harmless: addresses are interchangeable, and
   the permutation lemma covers which instance takes which one.
2. **Static channels are fresh by being outside the translation's image, not by size**
   (deviates from Req. 5.31). `r* = @k(@0)` and `f* = @k(@k(@0))` are never encodings of
   source names, and no (o5) chain builds a `k` node. A size condition cannot hold for a
   channel shared by all programs without breaking canonicity. The check asserts the
   property that matters: no static channel occurs as a value.
3. **Hole indices are de Bruijn over contexts** and are kept on import to the matrix
   machine. Mat v0.5 Req. 3.3 instead collapses them to one `HOLE`.
4. **The marking analysis is reported, not acted on.** Every unit is erected by `inst`,
   including those marked safe for the fixed-arity family, so Req. 5.17's "MUST compile by
   the fixed-arity family" is not followed. `inst` is correct on all units, and the
   fixed-arity path would need the literal gadget machinery the spec forbids elsewhere.
5. **(o5) chains** are the cons_|/cons_m chains of Prop. 3.12 for quotations with no input.
   A quotation containing both an input and a bound name is `comb-o5-unsupported`. The
   previous build rebuilt it with positional apparatus, which the two-phase design shows to
   be unsound.
6. **`cstar` has provisional tag `0x63`.**
7. **Under the curried scheme the gates' `b` are static**, which is what `phase2.py`'s E6
   does. The `inst` build keeps `b` bound, so the two schemes' erected gates differ in `b`
   alone (Obl. 11.6 compares up to it).
8. **The adversarial scheduler is a host resolver.** It reads a row's instance off its
   names: the leaf's address is its spine minus the last Elias-gamma code. It needs the
   intern table, so the device does not run it.

## Findings for the papers and specs

1. **Collapsing hole indices is unsound for nested units** (Mat v0.5 Req. 3.3 and §13 item 6;
   the single `hole` of `rhocombmat.py`). A nested unit's context sits inside its parent's
   context. The parent's leaves in it must be filled by the parent's `inst`; the child's
   own hole must survive until the child is instantiated. With one hole marker, the
   parent's `inst` also fills the child's hole with the parent's address. This happens
   whenever an input's body has an input that mentions the outer bound name, e.g.
   `for(y<-a){ for(z<-b){ y!(*z) } }`. De Bruijn indices over context nodes fix it, and so
   does v0.6's "hole carrying its index" if the index is read that way. `phase2.py` never
   nests, so it does not see the problem.
2. **Address supply for received code is unspecified.** Req. 5.11 provides spares for nested
   units, and Ex. 6.9 relies on the unit re-posting at its own `r`. Nothing supplies an
   address when a drop runs code that arrived as a message and has its own inputs, as W's
   `*z | *z` does. Per-release-site supply at one shared `r*` (decision 1) closes this gap.
   The paper should state an equivalent.
3. **Phase two erects quoted processes too, not only input bodies.** A quoted process with
   an input is a unit (Req. 5.7 reads "the body of each input"). Its image is what a drop
   instantiates, and its canonical form is what `[[@P]]` must denote.
4. **Carried from earlier builds, still open in draft 3:**
   - Ex. 6.9 / Table 1 (o5): a chain's result is a message, so a subject position needs an
     adapter (`br`, `bl`) and a payload position needs a relay.
   - Links that act on public names must be placed inside the innermost enclosing input.
   - (o3) against a dynamic channel needs a proxy.
5. **The curried build does not reach W.** W's gate stores `e(c1) | e(c2)`, two scoped
   atoms, so Mat v0.5's curried-stores hazard applies to the paper's own regression term.
   On the corpus, curried reaches 12 of 21 programs. Five of the misses have a different
   cause: an inner input mentions an outer bound name, as in
   `for(y<-a){ for(z<-b){ y!(*z) } }`. The curried template has one hole, so it has no way
   to refer to a parent unit's names. A logic read over the curried build (Req. 5.27)
   therefore sees neither class of program.
6. **Marking must exclude distributors.** `d(q1, p1, p2)` mentions two flows but splits one
   message; counting it would mark the near miss unsafe.

## Not implemented

* The fixed-arity path for safe units (decision 4). Presentation B.
* Shortcuts (Rem. 6.4).
* The CampF1R3-backed machine (`f1r3comb-space`, `f1r3comb-run`). All runs use the
  F1R3Comb-Mat machine.
* `wasm32` builds and the performance targets.
