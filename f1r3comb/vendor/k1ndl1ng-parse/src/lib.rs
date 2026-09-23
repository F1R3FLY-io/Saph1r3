//! `k1ndl1ng-parse` — text to a typed tree, and nothing else.
//!
//! The parser has no opinion about binding (Req. 6.1): unbound, duplicated and
//! shadowed variables all parse. It does not reorder a parallel composition and
//! does not normalise (Req. 3.4). Every traversal is driven by an explicit work
//! stack on the heap; no function recurses on term structure (Req. 4.5), so a
//! gesture interface cannot overflow the browser main thread's stack with
//! nested quotes. Levels are a parser parameter (Req. 2.1).

#![forbid(unsafe_code)]

pub mod lex;

use k1ndl1ng_ast::*;
use lex::{lex, Tok, Token};

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub enum Level {
    K0,
    K1,
    K2,
}

impl Level {
    fn name(self) -> &'static str {
        match self {
            Level::K0 => "K0",
            Level::K1 => "K1",
            Level::K2 => "K2",
        }
    }
}

#[derive(Clone, Debug)]
pub struct Options {
    pub level: Level,
    pub max_depth: u32,
    pub desugar_seq: bool,
}

impl Default for Options {
    fn default() -> Self {
        Options {
            level: Level::K1,
            max_depth: 1 << 16,
            desugar_seq: true,
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Severity {
    Error,
    Warning,
}

#[derive(Clone, Debug)]
pub struct Diag {
    pub span: Span,
    pub severity: Severity,
    pub code: &'static str,
    pub message: String,
}

impl Diag {
    fn error(span: Span, code: &'static str, message: String) -> Diag {
        Diag {
            span,
            severity: Severity::Error,
            code,
            message,
        }
    }
}

impl std::fmt::Display for Diag {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}..{}: {} [{}]",
            self.span.lo, self.span.hi, self.message, self.code
        )
    }
}

pub struct Parsed {
    pub tree: Ast,
    pub diags: Vec<Diag>,
}

impl Parsed {
    pub fn ok(&self) -> bool {
        !self.diags.iter().any(|d| d.severity == Severity::Error)
    }
}

/// Parse a whole source text.
pub fn parse(src: &str, opts: &Options) -> Parsed {
    Parser::new(src, opts).run(true)
}

/// Parse one process, for splicing into an existing tree (spec §10.2).
pub fn parse_fragment(src: &str, opts: &Options) -> Parsed {
    Parser::new(src, opts).run(false)
}

// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq)]
enum Closer {
    Paren,
    Brace,
}

struct BindSpec {
    kind: BindKind,
    pats: Vec<NameId>,
    chan: NameId,
}

struct RecvState {
    lo: u32,
    done: Vec<Vec<BindSpec>>,
    cur: Vec<BindSpec>,
    pats: Vec<NameId>,
    kind: BindKind,
}

enum Frame {
    Par { parts: Vec<ProcId>, lo: u32 },
    Group { closer: Closer },
    Quote { lo: u32 },
    Eval { lo: u32 },
    SendHead { lo: u32 },
    SendArgs { chan: NameId, persistent: bool, args: Vec<ProcId>, lo: u32 },
    NewBody { ids: Vec<Sym>, lo: u32 },
    RecvPat { st: RecvState },
    RecvChan { st: RecvState },
    RecvBody { st: RecvState },
}

enum Mode {
    NeedProc,
    NeedTerm,
    NeedQuoted,
    NeedName,
    HaveProc(ProcId),
    HaveName(NameId),
}

struct Parser<'a> {
    src: &'a str,
    toks: Vec<Token>,
    pos: usize,
    b: Builder,
    diags: Vec<Diag>,
    opts: Options,
    depth_exceeded: bool,
}

impl<'a> Parser<'a> {
    fn new(src: &'a str, opts: &Options) -> Parser<'a> {
        let lexed = lex(src);
        let mut b = Builder::new();
        for t in &lexed.trivia {
            b.add_trivia(*t);
        }
        let diags = lexed
            .bad
            .iter()
            .map(|(s, m)| Diag::error(*s, "lex", (*m).to_string()))
            .collect();
        Parser {
            src,
            toks: lexed.toks,
            pos: 0,
            b,
            diags,
            opts: opts.clone(),
            depth_exceeded: false,
        }
    }

    fn peek(&self) -> Token {
        self.toks[self.pos.min(self.toks.len() - 1)]
    }
    fn peek2(&self) -> Token {
        self.toks[(self.pos + 1).min(self.toks.len() - 1)]
    }
    fn at(&self, k: Tok) -> bool {
        self.peek().kind == k
    }
    fn bump(&mut self) -> Token {
        let t = self.peek();
        if self.pos + 1 < self.toks.len() {
            self.pos += 1;
        }
        t
    }
    fn eat(&mut self, k: Tok) -> bool {
        if self.at(k) {
            self.bump();
            true
        } else {
            false
        }
    }
    fn text(&self, t: Token) -> &'a str {
        &self.src[t.lo as usize..t.hi as usize]
    }
    fn err(&mut self, span: Span, code: &'static str, msg: String) {
        if self.diags.len() < 256 {
            self.diags.push(Diag::error(span, code, msg));
        }
    }
    /// Report a construct that a higher level would accept (Req. 2.1).
    fn need_level(&mut self, span: Span, what: &str, lvl: Level) {
        let msg = format!(
            "{} needs {}; this parser is at {}",
            what,
            lvl.name(),
            self.opts.level.name()
        );
        self.err(span, "level", msg);
    }
    fn expect(&mut self, k: Tok, what: &str) -> bool {
        if self.eat(k) {
            true
        } else {
            let t = self.peek();
            let found = if t.kind == Tok::Eof {
                "end of input".to_string()
            } else {
                format!("`{}`", self.text(t))
            };
            self.err(
                t.span(),
                "expected",
                format!("expected {what}, found {found}"),
            );
            false
        }
    }
    fn error_proc(&mut self, span: Span) -> ProcId {
        let d = DiagId(self.diags.len().saturating_sub(1) as u32);
        self.b.add_proc(ProcNode::Error { diag: d }, span)
    }

    fn run(mut self, whole: bool) -> Parsed {
        let mut stack: Vec<Frame> = Vec::new();
        let mut mode = Mode::NeedProc;
        let root;
        loop {
            if stack.len() as u32 > self.opts.max_depth {
                if !self.depth_exceeded {
                    self.depth_exceeded = true;
                    let s = self.peek().span();
                    self.err(
                        s,
                        "depth",
                        format!("nesting deeper than max_depth ({})", self.opts.max_depth),
                    );
                }
                // Unwind without consuming more input.
                let p = self.error_proc(self.peek().span());
                stack.clear();
                root = p;
                break;
            }
            match mode {
                Mode::NeedProc => {
                    stack.push(Frame::Par {
                        parts: Vec::new(),
                        lo: self.peek().lo,
                    });
                    mode = Mode::NeedTerm;
                }
                Mode::NeedTerm | Mode::NeedQuoted => {
                    let quoted = matches!(mode, Mode::NeedQuoted);
                    let t = self.peek();
                    match t.kind {
                        Tok::Nil => {
                            self.bump();
                            let p = self.b.add_proc(ProcNode::Nil, t.span());
                            mode = Mode::HaveProc(p);
                        }
                        Tok::LParen => {
                            self.bump();
                            stack.push(Frame::Group {
                                closer: Closer::Paren,
                            });
                            mode = Mode::NeedProc;
                        }
                        Tok::LBrace => {
                            self.bump();
                            stack.push(Frame::Group {
                                closer: Closer::Brace,
                            });
                            mode = Mode::NeedProc;
                        }
                        Tok::Star => {
                            self.bump();
                            stack.push(Frame::Eval { lo: t.lo });
                            mode = Mode::NeedName;
                        }
                        Tok::At if quoted => {
                            self.bump();
                            stack.push(Frame::Quote { lo: t.lo });
                            mode = Mode::NeedQuoted;
                        }
                        Tok::Ident if quoted => {
                            self.bump();
                            let s = self.b.intern(self.text(t));
                            let p = self.b.add_proc(ProcNode::Var { id: s }, t.span());
                            mode = Mode::HaveProc(p);
                        }
                        Tok::Wild if quoted => {
                            self.bump();
                            if self.opts.level < Level::K1 {
                                self.need_level(t.span(), "the wildcard `_`", Level::K1);
                            }
                            let p = self.b.add_proc(ProcNode::Wild, t.span());
                            mode = Mode::HaveProc(p);
                        }
                        Tok::At => {
                            // A name in process position: the head of a send.
                            stack.push(Frame::SendHead { lo: t.lo });
                            mode = Mode::NeedName;
                        }
                        Tok::Ident | Tok::Wild => {
                            // One token of lookahead decides: a name followed by
                            // `!` heads a send; otherwise this is a process
                            // variable (or a wildcard), which is what a bound
                            // `@P` pattern variable looks like in the body.
                            let nx = self.peek2().kind;
                            if nx == Tok::Bang || nx == Tok::BangBang {
                                stack.push(Frame::SendHead { lo: t.lo });
                                mode = Mode::NeedName;
                            } else {
                                self.bump();
                                let p = if t.kind == Tok::Wild {
                                    if self.opts.level < Level::K1 {
                                        self.need_level(t.span(), "the wildcard `_`", Level::K1);
                                    }
                                    self.b.add_proc(ProcNode::Wild, t.span())
                                } else {
                                    let sym = self.b.intern(self.text(t));
                                    self.b.add_proc(ProcNode::Var { id: sym }, t.span())
                                };
                                mode = Mode::HaveProc(p);
                            }
                        }
                        Tok::For => {
                            self.bump();
                            self.expect(Tok::LParen, "`(` after `for`");
                            stack.push(Frame::RecvPat {
                                st: RecvState {
                                    lo: t.lo,
                                    done: Vec::new(),
                                    cur: Vec::new(),
                                    pats: Vec::new(),
                                    kind: BindKind::Linear,
                                },
                            });
                            mode = Mode::NeedName;
                        }
                        Tok::New => {
                            self.bump();
                            if self.opts.level < Level::K1 {
                                self.need_level(t.span(), "`new`", Level::K1);
                            }
                            let mut ids = Vec::new();
                            loop {
                                let it = self.peek();
                                if it.kind == Tok::Ident {
                                    self.bump();
                                    let s = self.b.intern(self.text(it));
                                    ids.push(s);
                                } else {
                                    self.err(
                                        it.span(),
                                        "expected",
                                        "expected an identifier in `new`".to_string(),
                                    );
                                }
                                if !self.eat(Tok::Comma) {
                                    break;
                                }
                            }
                            self.expect(Tok::In, "`in`");
                            self.expect(Tok::LBrace, "`{`");
                            stack.push(Frame::NewBody { ids, lo: t.lo });
                            mode = Mode::NeedProc;
                        }
                        Tok::Reserved | Tok::Where | Tok::In => {
                            self.bump();
                            let w = self.text(t).to_string();
                            self.err(
                                t.span(),
                                "not-kernel",
                                format!("`{w}` is reserved and is not in this kernel"),
                            );
                            let p = self.error_proc(t.span());
                            mode = Mode::HaveProc(p);
                        }
                        _ => {
                            let found = if t.kind == Tok::Eof {
                                "end of input".to_string()
                            } else {
                                format!("`{}`", self.text(t))
                            };
                            self.err(
                                t.span(),
                                "expected",
                                format!("expected a process, found {found}"),
                            );
                            if t.kind != Tok::Eof
                                && !matches!(t.kind, Tok::RParen | Tok::RBrace | Tok::Comma | Tok::Bar)
                            {
                                self.bump();
                            }
                            let p = self.error_proc(t.span());
                            mode = Mode::HaveProc(p);
                        }
                    }
                }
                Mode::NeedName => {
                    let t = self.peek();
                    match t.kind {
                        Tok::Ident => {
                            self.bump();
                            let s = self.b.intern(self.text(t));
                            let n = self.b.add_name(NameNode::Var { id: s }, t.span());
                            mode = Mode::HaveName(n);
                        }
                        Tok::Wild => {
                            self.bump();
                            if self.opts.level < Level::K1 {
                                self.need_level(t.span(), "the wildcard `_`", Level::K1);
                            }
                            let n = self.b.add_name(NameNode::Wild, t.span());
                            mode = Mode::HaveName(n);
                        }
                        Tok::At => {
                            self.bump();
                            stack.push(Frame::Quote { lo: t.lo });
                            mode = Mode::NeedQuoted;
                        }
                        _ => {
                            let found = if t.kind == Tok::Eof {
                                "end of input".to_string()
                            } else {
                                format!("`{}`", self.text(t))
                            };
                            self.err(
                                t.span(),
                                "expected",
                                format!("expected a name, found {found}"),
                            );
                            if t.kind != Tok::Eof
                                && !matches!(t.kind, Tok::RParen | Tok::RBrace | Tok::Comma)
                            {
                                self.bump();
                            }
                            let p = self.error_proc(t.span());
                            let n = self.b.add_name(NameNode::Quote(p), t.span());
                            mode = Mode::HaveName(n);
                        }
                    }
                }
                Mode::HaveName(n) => {
                    let frame = match stack.pop() {
                        Some(f) => f,
                        None => {
                            // A bare name is not a process.
                            let sp = self.b_name_span(n);
                            self.err(sp, "expected", "expected a process".to_string());
                            root = self.error_proc(sp);
                            break;
                        }
                    };
                    match frame {
                        Frame::Eval { lo } => {
                            let hi = self.prev_hi();
                            let p = self.b.add_proc(ProcNode::Eval { name: n }, Span::new(lo, hi));
                            mode = Mode::HaveProc(p);
                        }
                        Frame::SendHead { lo } => {
                            let t = self.peek();
                            let persistent = match t.kind {
                                Tok::Bang => {
                                    self.bump();
                                    false
                                }
                                Tok::BangBang => {
                                    self.bump();
                                    if self.opts.level < Level::K1 {
                                        self.need_level(t.span(), "persistent send `!!`", Level::K1);
                                    }
                                    true
                                }
                                _ => {
                                    self.err(
                                        t.span(),
                                        "expected",
                                        "expected `!` or `!!` after a channel".to_string(),
                                    );
                                    false
                                }
                            };
                            self.expect(Tok::LParen, "`(`");
                            if self.eat(Tok::RParen) {
                                if self.opts.level < Level::K1 {
                                    self.need_level(
                                        Span::new(lo, self.prev_hi()),
                                        "a send of arity other than one",
                                        Level::K1,
                                    );
                                }
                                let p = self.b.send(n, persistent, &[]);
                                mode = Mode::HaveProc(p);
                            } else {
                                stack.push(Frame::SendArgs {
                                    chan: n,
                                    persistent,
                                    args: Vec::new(),
                                    lo,
                                });
                                mode = Mode::NeedProc;
                            }
                        }
                        Frame::RecvPat { mut st } => {
                            st.pats.push(n);
                            let t = self.peek();
                            match t.kind {
                                Tok::Comma => {
                                    self.bump();
                                    if self.opts.level < Level::K1 {
                                        self.need_level(t.span(), "a polyadic receipt", Level::K1);
                                    }
                                    stack.push(Frame::RecvPat { st });
                                    mode = Mode::NeedName;
                                }
                                Tok::LArrow | Tok::LEq | Tok::LPeek => {
                                    self.bump();
                                    st.kind = match t.kind {
                                        Tok::LArrow => BindKind::Linear,
                                        Tok::LEq => BindKind::Persistent,
                                        _ => BindKind::Peek,
                                    };
                                    if st.kind != BindKind::Linear && self.opts.level < Level::K1 {
                                        self.need_level(
                                            t.span(),
                                            match st.kind {
                                                BindKind::Persistent => "persistent receive `<=`",
                                                _ => "peek `<<-`",
                                            },
                                            Level::K1,
                                        );
                                    }
                                    stack.push(Frame::RecvChan { st });
                                    mode = Mode::NeedName;
                                }
                                _ => {
                                    self.err(
                                        t.span(),
                                        "expected",
                                        "expected `<-`, `<=`, `<<-` or `,` in a receipt".to_string(),
                                    );
                                    stack.push(Frame::RecvChan { st });
                                    mode = Mode::NeedName;
                                }
                            }
                        }
                        Frame::RecvChan { mut st } => {
                            let pats = std::mem::take(&mut st.pats);
                            st.cur.push(BindSpec {
                                kind: st.kind,
                                pats,
                                chan: n,
                            });
                            let t = self.peek();
                            match t.kind {
                                Tok::Amp => {
                                    self.bump();
                                    if self.opts.level < Level::K1 {
                                        self.need_level(t.span(), "a join `&`", Level::K1);
                                    }
                                    stack.push(Frame::RecvPat { st });
                                    mode = Mode::NeedName;
                                }
                                Tok::Semi => {
                                    self.bump();
                                    if self.opts.level < Level::K1 {
                                        self.need_level(
                                            t.span(),
                                            "sequential receipts `;`",
                                            Level::K1,
                                        );
                                    }
                                    let cur = std::mem::take(&mut st.cur);
                                    st.done.push(cur);
                                    stack.push(Frame::RecvPat { st });
                                    mode = Mode::NeedName;
                                }
                                Tok::Where => {
                                    self.bump();
                                    self.need_level(t.span(), "a `where` guard", Level::K2);
                                    // Skip the guard expression to the closing paren.
                                    let mut d = 0i32;
                                    loop {
                                        let u = self.peek();
                                        if u.kind == Tok::Eof {
                                            break;
                                        }
                                        if u.kind == Tok::LParen {
                                            d += 1;
                                        }
                                        if u.kind == Tok::RParen {
                                            if d == 0 {
                                                break;
                                            }
                                            d -= 1;
                                        }
                                        self.bump();
                                    }
                                    stack.push(Frame::RecvChan { st });
                                    mode = Mode::HaveName(n);
                                }
                                _ => {
                                    self.expect(Tok::RParen, "`)` closing the receipt");
                                    self.expect(Tok::LBrace, "`{`");
                                    let cur = std::mem::take(&mut st.cur);
                                    st.done.push(cur);
                                    stack.push(Frame::RecvBody { st });
                                    mode = Mode::NeedProc;
                                }
                            }
                        }
                        Frame::Quote { .. }
                        | Frame::Par { .. }
                        | Frame::Group { .. }
                        | Frame::SendArgs { .. }
                        | Frame::NewBody { .. }
                        | Frame::RecvBody { .. } => {
                            // Unreachable by construction: these frames resolve on
                            // a process, never on a name. Recover conservatively.
                            stack.push(frame);
                            let sp = self.b_name_span(n);
                            let p = self.error_proc(sp);
                            mode = Mode::HaveProc(p);
                        }
                    }
                }
                Mode::HaveProc(p) => {
                    let frame = match stack.pop() {
                        Some(f) => f,
                        None => {
                            root = p;
                            break;
                        }
                    };
                    match frame {
                        Frame::Par { mut parts, lo } => {
                            parts.push(p);
                            if self.at(Tok::Bar) {
                                self.bump();
                                stack.push(Frame::Par { parts, lo });
                                mode = Mode::NeedTerm;
                            } else if parts.len() == 1 {
                                mode = Mode::HaveProc(parts[0]);
                            } else {
                                let s = self.b.proc_slice(&parts);
                                let hi = self.prev_hi();
                                let q = self.b.add_proc(ProcNode::Par(s), Span::new(lo, hi));
                                mode = Mode::HaveProc(q);
                            }
                        }
                        Frame::Group { closer } => {
                            match closer {
                                Closer::Paren => self.expect(Tok::RParen, "`)`"),
                                Closer::Brace => self.expect(Tok::RBrace, "`}`"),
                            };
                            mode = Mode::HaveProc(p);
                        }
                        Frame::Quote { lo } => {
                            let hi = self.prev_hi();
                            let n = self.b.add_name(NameNode::Quote(p), Span::new(lo, hi));
                            mode = Mode::HaveName(n);
                        }
                        Frame::SendArgs {
                            chan,
                            persistent,
                            mut args,
                            lo,
                        } => {
                            args.push(p);
                            if self.at(Tok::Comma) {
                                self.bump();
                                if self.opts.level < Level::K1 {
                                    self.need_level(
                                        self.peek().span(),
                                        "a polyadic send",
                                        Level::K1,
                                    );
                                }
                                stack.push(Frame::SendArgs {
                                    chan,
                                    persistent,
                                    args,
                                    lo,
                                });
                                mode = Mode::NeedProc;
                            } else {
                                self.expect(Tok::RParen, "`)` closing the arguments");
                                let s = self.b.proc_slice(&args);
                                let hi = self.prev_hi();
                                let q = self.b.add_proc(
                                    ProcNode::Send {
                                        chan,
                                        persistent,
                                        args: s,
                                    },
                                    Span::new(lo, hi),
                                );
                                mode = Mode::HaveProc(q);
                            }
                        }
                        Frame::NewBody { ids, lo } => {
                            self.expect(Tok::RBrace, "`}` closing the `new` body");
                            let s = self.b.sym_slice(&ids);
                            let hi = self.prev_hi();
                            let q = self
                                .b
                                .add_proc(ProcNode::New { names: s, body: p }, Span::new(lo, hi));
                            mode = Mode::HaveProc(q);
                        }
                        Frame::RecvBody { st } => {
                            self.expect(Tok::RBrace, "`}` closing the receive body");
                            let hi = self.prev_hi();
                            let q = self.build_receive(st, p, hi);
                            mode = Mode::HaveProc(q);
                        }
                        Frame::Eval { .. } | Frame::SendHead { .. } | Frame::RecvPat { .. }
                        | Frame::RecvChan { .. } => {
                            stack.push(frame);
                            let n = self.b.add_name(NameNode::Quote(p), Span::NULL);
                            mode = Mode::HaveName(n);
                        }
                    }
                }
            }
        }

        if whole && !self.at(Tok::Eof) {
            let t = self.peek();
            let txt = self.text(t).to_string();
            self.err(
                t.span(),
                "trailing",
                format!("unexpected `{txt}` after the end of the process"),
            );
        }
        let tree = self.b.finish(root);
        Parsed {
            tree,
            diags: self.diags,
        }
    }

    fn b_name_span(&self, _n: NameId) -> Span {
        self.peek().span()
    }

    fn prev_hi(&self) -> u32 {
        if self.pos == 0 {
            0
        } else {
            self.toks[self.pos - 1].hi
        }
    }

    /// Build the receive, desugaring sequential receipts right to left:
    /// `for(a; b){P}` is `for(a){for(b){P}}`.
    fn build_receive(&mut self, st: RecvState, body: ProcId, hi: u32) -> ProcId {
        let mut receipts = st.done;
        if receipts.is_empty() {
            receipts.push(Vec::new());
        }
        if receipts.len() > 1 && !self.opts.desugar_seq {
            self.err(
                Span::new(st.lo, hi),
                "seq",
                "sequential receipts are disabled (desugar_seq = false)".to_string(),
            );
        }
        let mut inner = body;
        for group in receipts.into_iter().rev() {
            let mut first = None;
            let mut count = 0u32;
            for bs in &group {
                let pats = self.b.name_slice(&bs.pats);
                let id = self.b.add_bind(Bind {
                    kind: bs.kind,
                    pats,
                    chan: bs.chan,
                });
                if first.is_none() {
                    first = Some(id);
                }
                count += 1;
            }
            let r = self.b.add_receipt(first.unwrap_or(BindId(0)), count, None);
            inner = self.b.add_proc(
                ProcNode::Receive {
                    receipt: r,
                    body: inner,
                },
                Span::new(st.lo, hi),
            );
        }
        inner
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use k1ndl1ng_ast::print::{print, Style};

    fn p(src: &str) -> Parsed {
        parse(src, &Options::default())
    }

    #[test]
    fn round_trip_simple() {
        let r = p("x!(Nil) | for(y <- x) { *y }");
        assert!(r.ok(), "{:?}", r.diags);
        assert_eq!(r.tree.check(), Ok(()));
        let s = print(&r.tree, Style::Compact);
        assert_eq!(s, "x!(Nil) | for(y <- x) {*y}");
    }

    #[test]
    fn par_is_flat() {
        let r = p("Nil | Nil | Nil");
        assert!(r.ok());
        match r.tree.proc(r.tree.root()) {
            ProcNode::Par(s) => assert_eq!(s.len, 3),
            other => panic!("expected a flat par, got {other:?}"),
        }
    }

    #[test]
    fn quotes_and_unquotes() {
        let r = p("@{Nil}!(*@{x!(Nil)})");
        assert!(r.ok(), "{:?}", r.diags);
        assert_eq!(r.tree.check(), Ok(()));
    }

    #[test]
    fn joins_peeks_persistence() {
        let r = p("for(x <- a & y <<- b) { Nil } | for(@z <= c) { c!!(z) }");
        assert!(r.ok(), "{:?}", r.diags);
        assert_eq!(r.tree.check(), Ok(()));
    }

    #[test]
    fn sequential_receipts_desugar() {
        let r = p("for(x <- a; y <- b) { Nil }");
        assert!(r.ok(), "{:?}", r.diags);
        let s = print(&r.tree, Style::Compact);
        assert_eq!(s, "for(x <- a) {for(y <- b) {Nil}}");
    }

    #[test]
    fn k0_rejects_k1_forms() {
        let opts = Options {
            level: Level::K0,
            ..Default::default()
        };
        for src in ["x!!(Nil)", "for(y <= x){Nil}", "new a in { Nil }", "x!(Nil, Nil)"] {
            let r = parse(src, &opts);
            assert!(!r.ok(), "{src} should not parse at K0");
            assert!(r.diags.iter().any(|d| d.code == "level"), "{src}: {:?}", r.diags);
        }
    }

    #[test]
    fn reserved_words_are_not_identifiers() {
        let r = p("contract!(Nil)");
        assert!(!r.ok());
        assert!(r.diags.iter().any(|d| d.code == "not-kernel"));
    }

    #[test]
    fn deep_nesting_does_not_overflow() {
        let n = 20000;
        let mut s = String::new();
        for _ in 0..n {
            s.push_str("@{");
        }
        s.push_str("Nil");
        for _ in 0..n {
            s.push('}');
        }
        s.push_str("!(Nil)");
        let r = p(&s);
        // Either it parses, or it reports max_depth. It must not abort.
        assert!(r.ok() || r.diags.iter().any(|d| d.code == "depth"));
    }

    #[test]
    fn garbage_never_panics() {
        for src in ["", "(((((", "for(", "@", "x!", "\u{1F600}", "}{)(", "for(x <- ) {"] {
            let _ = p(src);
        }
    }
}
