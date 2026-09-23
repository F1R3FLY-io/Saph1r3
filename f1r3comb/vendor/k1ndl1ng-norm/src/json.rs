//! The structural form of a term, as JSON, and a small writer to build it
//! with.
//!
//! A normal form has a printed form (`show`) and an encoded form
//! (`encode`), and neither is what a renderer wants. A printer gives text a
//! person reads; the encoding gives bytes the store keys by. A renderer — a
//! headset drawing a robot per constructor, a browser drawing a tree, an LSP
//! server answering a hover — wants the *shape*: the constructors, their
//! children, and enough identity on each to follow an object across frames.
//!
//! This is specified here, beside the normaliser, for the reason §11 of the
//! k1ndl1ng spec gives for putting the editor and scene services here rather
//! than in any client: so that a text editor, an LSP server and a headset
//! share one implementation. Nothing in this module knows what a rhobot is.
//!
//! Three properties are carried deliberately.
//!
//! **Every node carries its fingerprint.** `fp` is the content hash of the
//! subterm. It is the same 32 bytes the store keys by, so two names that
//! agree here are equivalent, and a client never has to decide name
//! equivalence itself.
//!
//! **Elision happens here, not in the client.** A consumer says how deep it
//! can still draw a quotation; anything past that comes back as `elided`
//! *with its fingerprint intact*, so an elided name stays comparable. Without
//! it a single deep term sends megabytes to a consumer that would draw three
//! levels of it.
//!
//! **The walk does not recurse.** Req. 4.5 forbids recursion on term
//! structure anywhere, on the grounds that a gesture interface nests quotes by
//! accident. This walks an explicit stack, like every other traversal in the
//! crate: each node builds its output in reading order and pushes it
//! reversed, so the text is written left to right by a loop that never nests.

use crate::term::{BindKind, Name, Node, Norm};

// ---------------------------------------------------------------------------
// A very small JSON writer.
//
// The crate has no dependencies and this does not change that. It is a writer
// only: nothing here parses JSON.

/// A growable JSON output buffer.
pub struct Buf(String);

impl Default for Buf {
    fn default() -> Self {
        Buf::new()
    }
}

impl Buf {
    pub fn new() -> Buf {
        Buf(String::with_capacity(1024))
    }
    pub fn with_capacity(n: usize) -> Buf {
        Buf(String::with_capacity(n))
    }
    /// Append text that is already valid JSON, or is structural punctuation.
    pub fn raw(&mut self, s: &str) {
        self.0.push_str(s);
    }
    /// Append a quoted, escaped string.
    pub fn str(&mut self, s: &str) {
        self.0.push('"');
        self.0.push_str(&esc(s));
        self.0.push('"');
    }
    pub fn num(&mut self, n: u64) {
        self.0.push_str(&n.to_string());
    }
    pub fn bool(&mut self, b: bool) {
        self.0.push_str(if b { "true" } else { "false" });
    }
    /// `"key":` — the caller writes the value.
    pub fn key(&mut self, k: &str) {
        self.0.push('"');
        self.0.push_str(k);
        self.0.push_str("\":");
    }
    pub fn field_str(&mut self, k: &str, v: &str) {
        self.key(k);
        self.str(v);
    }
    pub fn field_num(&mut self, k: &str, v: u64) {
        self.key(k);
        self.num(v);
    }
    pub fn field_bool(&mut self, k: &str, v: bool) {
        self.key(k);
        self.bool(v);
    }
    /// `"key":<already-JSON>`
    pub fn field_raw(&mut self, k: &str, v: &str) {
        self.key(k);
        self.raw(v);
    }
    pub fn comma(&mut self) {
        self.0.push(',');
    }
    pub fn len(&self) -> usize {
        self.0.len()
    }
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
    pub fn as_str(&self) -> &str {
        &self.0
    }
    pub fn into_string(self) -> String {
        self.0
    }
}

/// JSON string escaping.
pub fn esc(s: &str) -> String {
    let mut o = String::with_capacity(s.len() + 2);
    for c in s.chars() {
        match c {
            '"' => o.push_str("\\\""),
            '\\' => o.push_str("\\\\"),
            '\n' => o.push_str("\\n"),
            '\r' => o.push_str("\\r"),
            '\t' => o.push_str("\\t"),
            c if (c as u32) < 0x20 => o.push_str(&format!("\\u{:04x}", c as u32)),
            c => o.push(c),
        }
    }
    o
}

// ---------------------------------------------------------------------------
// The tree

/// How much of a term is written out before it becomes an elision glyph.
#[derive(Copy, Clone, Debug)]
pub struct Budget {
    /// Quotation levels written. This counts nesting inside quotes, not
    /// parallel nesting, because it is quotation that a renderer shrinks.
    pub quote_depth: u32,
    /// Nodes written for one term, whatever the depth: a backstop against a
    /// wide term rather than a deep one.
    pub nodes: u32,
}

impl Default for Budget {
    fn default() -> Self {
        Budget {
            quote_depth: 4,
            nodes: 4096,
        }
    }
}

impl Budget {
    pub fn with_quote_depth(d: u32) -> Budget {
        Budget {
            quote_depth: d.min(16),
            ..Budget::default()
        }
    }
}

/// Write `t` as a JSON tree.
pub fn write_tree(out: &mut Buf, t: &Norm, budget: Budget) {
    let mut w = Walk {
        out,
        left: budget.nodes,
        max_q: budget.quote_depth,
    };
    let mut stack: Vec<Task> = vec![Task::Proc(t.clone(), 0)];
    while let Some(task) = stack.pop() {
        match task {
            Task::Lit(s) => w.out.raw(s),
            Task::Owned(s) => w.out.raw(&s),
            Task::Proc(p, q) => w.proc(&p, q, &mut stack),
            Task::Name(n, q) => w.name(&n, q, &mut stack),
        }
    }
}

/// Convenience for a single term.
pub fn tree_to_string(t: &Norm, budget: Budget) -> String {
    let mut b = Buf::new();
    write_tree(&mut b, t, budget);
    b.into_string()
}

/// One item of work. The `u32` is the quotation depth the item sits at.
enum Task {
    Proc(Norm, u32),
    Name(Name, u32),
    Lit(&'static str),
    Owned(String),
}

/// Push a sequence written in reading order so that it pops in that order.
fn schedule(stack: &mut Vec<Task>, seq: Vec<Task>) {
    stack.extend(seq.into_iter().rev());
}

/// Append `items` with commas between and none after. Each item is itself a
/// sequence, because a bind is several tasks long.
fn commas(seq: &mut Vec<Task>, items: Vec<Vec<Task>>) {
    let n = items.len();
    for (i, item) in items.into_iter().enumerate() {
        seq.extend(item);
        if i + 1 < n {
            seq.push(Task::Lit(","));
        }
    }
}

struct Walk<'a> {
    out: &'a mut Buf,
    left: u32,
    max_q: u32,
}

impl Walk<'_> {
    /// True when this item must be replaced by an elision glyph.
    fn spent(&mut self, q: u32) -> bool {
        if self.left == 0 || q > self.max_q {
            return true;
        }
        self.left -= 1;
        false
    }

    /// The head every process node shares: what it is, its fingerprint, and
    /// the metrics a consumer sizes it by.
    fn head(&mut self, p: &Norm, form: &str) {
        self.out.raw("{\"f\":\"");
        self.out.raw(form);
        self.out.raw("\",\"fp\":\"");
        self.out.raw(&p.hash().hex());
        self.out.raw("\",\"size\":");
        self.out.num(p.size() as u64);
        self.out.raw(",\"depth\":");
        self.out.num(p.depth() as u64);
    }

    fn proc(&mut self, p: &Norm, q: u32, stack: &mut Vec<Task>) {
        if self.spent(q) {
            self.head(p, "elided");
            self.out.raw("}");
            return;
        }
        match p.node() {
            Node::Nil => {
                self.head(p, "nil");
                self.out.raw("}");
            }

            Node::Par(parts) => {
                // Components are in encoding order, which is stable across
                // frames, so a consumer can keep an object's identity across a
                // re-layout even though the order carries no meaning.
                self.head(p, "par");
                self.out.raw(",\"parts\":[");
                let mut seq = Vec::new();
                commas(
                    &mut seq,
                    parts
                        .iter()
                        .map(|c| vec![Task::Proc(c.clone(), q)])
                        .collect(),
                );
                seq.push(Task::Lit("]}"));
                schedule(stack, seq);
            }

            Node::Send {
                chan,
                persistent,
                args,
            } => {
                self.head(p, "send");
                self.out.raw(",\"persistent\":");
                self.out.raw(if *persistent { "true" } else { "false" });
                self.out.raw(",\"chan\":");
                let mut seq = vec![Task::Name(chan.clone(), q), Task::Lit(",\"args\":[")];
                commas(
                    &mut seq,
                    args.iter()
                        // An argument is what a quotation would hold: one
                        // level in, as far as the budget is concerned.
                        .map(|a| vec![Task::Proc(a.clone(), q + 1)])
                        .collect(),
                );
                seq.push(Task::Lit("]}"));
                schedule(stack, seq);
            }

            Node::Receive { binds, body } => {
                self.head(p, "receive");
                self.out.raw(",\"binds\":[");
                let mut seq = Vec::new();
                let binds: Vec<Vec<Task>> = binds
                    .iter()
                    .map(|b| {
                        let mut one = vec![
                            Task::Owned(format!(
                                "{{\"kind\":\"{}\",\"arrow\":\"{}\",\"binders\":{},\"chan\":",
                                kind_name(b.kind),
                                b.kind.arrow(),
                                b.binders
                            )),
                            Task::Name(b.chan.clone(), q),
                            Task::Lit(",\"pats\":["),
                        ];
                        commas(
                            &mut one,
                            b.pats
                                .iter()
                                .map(|x| vec![Task::Name(x.clone(), q)])
                                .collect(),
                        );
                        one.push(Task::Lit("]}"));
                        one
                    })
                    .collect();
                commas(&mut seq, binds);
                seq.push(Task::Lit("],\"body\":"));
                seq.push(Task::Proc(body.clone(), q));
                seq.push(Task::Lit("}"));
                schedule(stack, seq);
            }

            Node::New { count, body } => {
                self.head(p, "new");
                self.out.raw(",\"count\":");
                self.out.num(*count as u64);
                self.out.raw(",\"body\":");
                schedule(stack, vec![Task::Proc(body.clone(), q), Task::Lit("}")]);
            }

            Node::Eval(n) => {
                // `*x`: a name standing at a process position, waiting for
                // substitution to put a process there.
                self.head(p, "drop");
                self.out.raw(",\"name\":");
                schedule(stack, vec![Task::Name(n.clone(), q), Task::Lit("}")]);
            }

            Node::BoundVar(i) => {
                self.head(p, "var");
                self.out.raw(",\"bound\":true,\"index\":");
                self.out.num(*i as u64);
                self.out.raw("}");
            }

            Node::FreeVar(l) => {
                self.head(p, "var");
                self.out.raw(",\"bound\":false,\"index\":");
                self.out.num(*l as u64);
                self.out.raw("}");
            }

            Node::Wild => {
                self.head(p, "wild");
                self.out.raw("}");
            }
        }
    }

    fn name(&mut self, n: &Name, q: u32, stack: &mut Vec<Task>) {
        let fp = n.content_hash().hex();
        match n {
            Name::Quote(p) => {
                if self.spent(q) {
                    self.out.raw("{\"g\":\"quote\",\"elided\":true,\"fp\":\"");
                    self.out.raw(&fp);
                    self.out.raw("\",\"size\":");
                    self.out.num(p.size() as u64);
                    self.out.raw(",\"depth\":");
                    self.out.num(p.depth() as u64);
                    self.out.raw("}");
                    return;
                }
                self.out.raw("{\"g\":\"quote\",\"elided\":false,\"fp\":\"");
                self.out.raw(&fp);
                self.out.raw("\",\"proc\":");
                schedule(stack, vec![Task::Proc(p.clone(), q + 1), Task::Lit("}")]);
            }
            Name::Bound(i) => self.slot(&fp, true, *i),
            Name::Free(l) => self.slot(&fp, false, *l),
            Name::Unforgeable(b) => {
                // Named as the printer names it, so a channel that reads
                // `u_3375c662` in a label reads the same here.
                self.out.raw("{\"g\":\"unforgeable\",\"fp\":\"");
                self.out.raw(&fp);
                self.out.raw("\",\"label\":\"");
                self.out.raw(&format!(
                    "u_{:02x}{:02x}{:02x}{:02x}",
                    b[0], b[1], b[2], b[3]
                ));
                self.out.raw("\"}");
            }
        }
    }

    /// A name variable: a slot a substitution will fill.
    fn slot(&mut self, fp: &str, bound: bool, index: u32) {
        self.out.raw("{\"g\":\"slot\",\"bound\":");
        self.out.raw(if bound { "true" } else { "false" });
        self.out.raw(",\"index\":");
        self.out.num(index as u64);
        self.out.raw(",\"fp\":\"");
        self.out.raw(fp);
        self.out.raw("\",\"label\":\"");
        self.out.raw(if bound { "n" } else { "f" });
        self.out.num(index as u64);
        self.out.raw("\"}");
    }
}

fn kind_name(k: BindKind) -> &'static str {
    match k {
        BindKind::Linear => "linear",
        BindKind::Persistent => "persistent",
        BindKind::Peek => "peek",
    }
}
