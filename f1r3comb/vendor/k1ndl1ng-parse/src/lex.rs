//! One left-to-right pass, no backtracking, no regular expressions. Tokens
//! carry a kind and a byte range and nothing else; text is recovered by slicing
//! the source. Trivia goes to a side table (spec §6).

use k1ndl1ng_ast::{Span, Trivia, TriviaKind};

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Tok {
    Ident,
    Wild,
    Nil,
    For,
    New,
    In,
    Where,
    /// A reserved word for a construct outside the kernel.
    Reserved,
    LParen,
    RParen,
    LBrace,
    RBrace,
    At,
    Star,
    Bang,
    BangBang,
    Bar,
    Amp,
    Semi,
    Comma,
    LArrow,
    LEq,
    LPeek,
    Unknown,
    Eof,
}

#[derive(Clone, Copy, Debug)]
pub struct Token {
    pub kind: Tok,
    pub lo: u32,
    pub hi: u32,
}

impl Token {
    pub fn span(&self) -> Span {
        Span::new(self.lo, self.hi)
    }
}

pub struct Lexed {
    pub toks: Vec<Token>,
    pub trivia: Vec<Trivia>,
    /// Byte offsets of ill-formed input, reported by the parser as diagnostics.
    pub bad: Vec<(Span, &'static str)>,
}

const RESERVED: &[(&str, Tok)] = &[
    ("Nil", Tok::Nil),
    ("for", Tok::For),
    ("new", Tok::New),
    ("in", Tok::In),
    ("where", Tok::Where),
    ("contract", Tok::Reserved),
    ("match", Tok::Reserved),
    ("select", Tok::Reserved),
    ("let", Tok::Reserved),
    ("if", Tok::Reserved),
    ("else", Tok::Reserved),
    ("bundle", Tok::Reserved),
    ("bundle0", Tok::Reserved),
    ("bundle+", Tok::Reserved),
    ("bundle-", Tok::Reserved),
    ("true", Tok::Reserved),
    ("false", Tok::Reserved),
    ("not", Tok::Reserved),
    ("and", Tok::Reserved),
    ("or", Tok::Reserved),
    ("matches", Tok::Reserved),
];

pub fn lex(src: &str) -> Lexed {
    let b = src.as_bytes();
    let n = b.len();
    let mut i = 0usize;
    let mut toks = Vec::new();
    let mut trivia = Vec::new();
    let mut bad = Vec::new();

    macro_rules! push {
        ($k:expr, $lo:expr, $hi:expr) => {
            toks.push(Token {
                kind: $k,
                lo: $lo as u32,
                hi: $hi as u32,
            })
        };
    }

    while i < n {
        let c = b[i];
        match c {
            b' ' | b'\t' | b'\r' | b'\n' => {
                i += 1;
            }
            b'/' if i + 1 < n && b[i + 1] == b'/' => {
                let lo = i;
                while i < n && b[i] != b'\n' {
                    i += 1;
                }
                trivia.push(Trivia {
                    kind: TriviaKind::LineComment,
                    span: Span::new(lo as u32, i as u32),
                });
            }
            b'/' if i + 1 < n && b[i + 1] == b'*' => {
                let lo = i;
                i += 2;
                let mut closed = false;
                while i + 1 < n {
                    if b[i] == b'*' && b[i + 1] == b'/' {
                        i += 2;
                        closed = true;
                        break;
                    }
                    i += 1;
                }
                if !closed {
                    i = n;
                    bad.push((Span::new(lo as u32, n as u32), "unterminated block comment"));
                }
                trivia.push(Trivia {
                    kind: TriviaKind::BlockComment,
                    span: Span::new(lo as u32, i as u32),
                });
            }
            b'(' => {
                push!(Tok::LParen, i, i + 1);
                i += 1;
            }
            b')' => {
                push!(Tok::RParen, i, i + 1);
                i += 1;
            }
            b'{' => {
                push!(Tok::LBrace, i, i + 1);
                i += 1;
            }
            b'}' => {
                push!(Tok::RBrace, i, i + 1);
                i += 1;
            }
            b'@' => {
                push!(Tok::At, i, i + 1);
                i += 1;
            }
            b'*' => {
                push!(Tok::Star, i, i + 1);
                i += 1;
            }
            b'|' => {
                push!(Tok::Bar, i, i + 1);
                i += 1;
            }
            b'&' => {
                push!(Tok::Amp, i, i + 1);
                i += 1;
            }
            b';' => {
                push!(Tok::Semi, i, i + 1);
                i += 1;
            }
            b',' => {
                push!(Tok::Comma, i, i + 1);
                i += 1;
            }
            b'!' => {
                if i + 1 < n && b[i + 1] == b'!' {
                    push!(Tok::BangBang, i, i + 2);
                    i += 2;
                } else {
                    push!(Tok::Bang, i, i + 1);
                    i += 1;
                }
            }
            b'<' => {
                if i + 2 < n && b[i + 1] == b'<' && b[i + 2] == b'-' {
                    push!(Tok::LPeek, i, i + 3);
                    i += 3;
                } else if i + 1 < n && b[i + 1] == b'-' {
                    push!(Tok::LArrow, i, i + 2);
                    i += 2;
                } else if i + 1 < n && b[i + 1] == b'=' {
                    push!(Tok::LEq, i, i + 2);
                    i += 2;
                } else {
                    push!(Tok::Unknown, i, i + 1);
                    bad.push((Span::new(i as u32, i as u32 + 1), "stray '<'"));
                    i += 1;
                }
            }
            b'_' => {
                // `_` alone is the wildcard; `_foo` is an identifier.
                let lo = i;
                let mut j = i + 1;
                while j < n && is_ident_cont(b[j]) {
                    j += 1;
                }
                if j == i + 1 {
                    push!(Tok::Wild, lo, j);
                } else {
                    push!(Tok::Ident, lo, j);
                }
                i = j;
            }
            _ if c.is_ascii_alphabetic() => {
                let lo = i;
                let mut j = i;
                while j < n && is_ident_cont(b[j]) {
                    j += 1;
                }
                let word = &src[lo..j];
                let kind = RESERVED
                    .iter()
                    .find(|(w, _)| *w == word)
                    .map(|(_, k)| *k)
                    .unwrap_or(Tok::Ident);
                push!(kind, lo, j);
                i = j;
            }
            _ => {
                let lo = i;
                // Consume a whole UTF-8 sequence so spans stay on char boundaries.
                i += 1;
                while i < n && (b[i] & 0xC0) == 0x80 {
                    i += 1;
                }
                push!(Tok::Unknown, lo, i);
                bad.push((
                    Span::new(lo as u32, i as u32),
                    if c >= 0x80 {
                        "non-ASCII byte; identifiers are ASCII in v1"
                    } else {
                        "unexpected character"
                    },
                ));
            }
        }
    }
    push!(Tok::Eof, n, n);
    Lexed { toks, trivia, bad }
}

fn is_ident_cont(c: u8) -> bool {
    c.is_ascii_alphanumeric() || c == b'_' || c == b'\''
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn arrows_and_bangs() {
        let l = lex("x!!(Nil) for(y <<- z & w <= v)");
        let kinds: Vec<Tok> = l.toks.iter().map(|t| t.kind).collect();
        assert_eq!(kinds[0], Tok::Ident);
        assert_eq!(kinds[1], Tok::BangBang);
        assert!(kinds.contains(&Tok::LPeek));
        assert!(kinds.contains(&Tok::LEq));
        assert!(kinds.contains(&Tok::Amp));
    }

    #[test]
    fn comments_are_trivia() {
        let l = lex("Nil // tail\n/* block */ Nil");
        assert_eq!(l.trivia.len(), 2);
        assert_eq!(l.toks.len(), 3); // Nil Nil Eof
    }

    #[test]
    fn wildcard_versus_identifier() {
        let l = lex("_ _x");
        assert_eq!(l.toks[0].kind, Tok::Wild);
        assert_eq!(l.toks[1].kind, Tok::Ident);
    }
}
