//! The `.comb` artefact: the saved target (F1R3Comb v0.6 §8, step 5).
//!
//! Layout (integers little-endian or LEB128):
//!
//! ```text
//! magic          8 bytes  "F1R3COMB"
//! version        1 byte   2
//! presentation   1 byte   b'A'
//! shortcuts      1 byte   0
//! scheme         1 byte   1 = inst, 2 = curried, 0xEE = positional (test only)
//! source hash   32 bytes  BLAKE2b-256 of the K1ndl1ng normal form, or zeros
//! family         2 bytes  F(P): bit i set iff cons_A for base shape i is used
//! units          LEB128   number of units (quotations with bound names)
//! unsafe units   LEB128   units the marking analysis finds unsafe for the
//!                         fixed-arity family (Req. 5.17)
//! size bound     LEB128   encoding length of the largest name of the image;
//!                         a root address must exceed it (Req. 5.28)
//! length         LEB128   byte length of the term encoding
//! term           ...      canonical encoding
//! ```

use crate::{leb, read_leb, DecodeError, Hash32, Term};

pub const MAGIC: &[u8; 8] = b"F1R3COMB";
pub const VERSION: u8 = 2;
pub const PRESENTATION: u8 = b'A';
pub const SHORTCUTS: u8 = 0;

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Scheme {
    Inst,
    Curried,
    /// Positional apparatus: every instance shares its names. The
    /// construction of draft 3 §6.1; a negative control, never deployed.
    Positional,
}

impl Scheme {
    pub fn byte(self) -> u8 {
        match self {
            Scheme::Inst => 1,
            Scheme::Curried => 2,
            Scheme::Positional => 0xEE,
        }
    }
    pub fn from_byte(b: u8) -> Option<Scheme> {
        match b {
            1 => Some(Scheme::Inst),
            2 => Some(Scheme::Curried),
            0xEE => Some(Scheme::Positional),
            _ => None,
        }
    }
    pub fn name(self) -> &'static str {
        match self {
            Scheme::Inst => "inst",
            Scheme::Curried => "curried",
            Scheme::Positional => "positional (negative control)",
        }
    }
}

#[derive(Clone, Debug)]
pub struct Header {
    pub scheme: Scheme,
    pub source_hash: Hash32,
    pub family: u16,
    pub units: u64,
    pub unsafe_units: u64,
    pub size_bound: u64,
}

#[derive(Clone, Debug)]
pub struct Artefact {
    pub header: Header,
    pub term: Term,
}

#[derive(Clone, Debug)]
pub enum ArtefactError {
    Magic,
    Version(u8),
    PresentationMismatch(u8),
    FeatureMismatch(u8),
    SchemeUnknown(u8),
    Truncated,
    Term(DecodeError),
}

impl std::fmt::Display for ArtefactError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ArtefactError::Magic => write!(f, "not a .comb artefact (bad magic) [comb-decode]"),
            ArtefactError::Version(v) => write!(f, "unsupported .comb version {v} [comb-decode]"),
            ArtefactError::PresentationMismatch(p) => write!(
                f,
                "artefact was compiled under presentation {:?}; this build is presentation A [comb-presentation-mismatch]",
                *p as char
            ),
            ArtefactError::FeatureMismatch(s) => {
                write!(f, "artefact has shortcuts={s}; this build has shortcuts=0 [comb-feature-mismatch]")
            }
            ArtefactError::SchemeUnknown(s) => write!(f, "unknown erection scheme {s} [comb-scheme-mismatch]"),
            ArtefactError::Truncated => write!(f, "truncated .comb artefact [comb-decode]"),
            ArtefactError::Term(e) => write!(f, "{e}"),
        }
    }
}

impl Artefact {
    pub fn encode(&self) -> Vec<u8> {
        let h = &self.header;
        let enc = self.term.encode();
        let mut out = Vec::with_capacity(64 + enc.len());
        out.extend_from_slice(MAGIC);
        out.push(VERSION);
        out.push(PRESENTATION);
        out.push(SHORTCUTS);
        out.push(h.scheme.byte());
        out.extend_from_slice(&h.source_hash.0);
        out.extend_from_slice(&h.family.to_le_bytes());
        leb(h.units, &mut out);
        leb(h.unsafe_units, &mut out);
        leb(h.size_bound, &mut out);
        leb(enc.len() as u64, &mut out);
        out.extend_from_slice(enc);
        out
    }

    pub fn is_artefact(bytes: &[u8]) -> bool {
        bytes.len() >= 8 && &bytes[..8] == MAGIC
    }

    pub fn decode(bytes: &[u8]) -> Result<Artefact, ArtefactError> {
        if !Artefact::is_artefact(bytes) {
            return Err(ArtefactError::Magic);
        }
        if bytes.len() < 46 {
            return Err(ArtefactError::Truncated);
        }
        if bytes[8] != VERSION {
            return Err(ArtefactError::Version(bytes[8]));
        }
        if bytes[9] != PRESENTATION {
            return Err(ArtefactError::PresentationMismatch(bytes[9]));
        }
        if bytes[10] != SHORTCUTS {
            return Err(ArtefactError::FeatureMismatch(bytes[10]));
        }
        let scheme = Scheme::from_byte(bytes[11]).ok_or(ArtefactError::SchemeUnknown(bytes[11]))?;
        let mut h = [0u8; 32];
        h.copy_from_slice(&bytes[12..44]);
        let family = u16::from_le_bytes([bytes[44], bytes[45]]);
        let mut pos = 46usize;
        let mut rd = || read_leb(bytes, &mut pos).ok_or(ArtefactError::Truncated);
        let units = rd()?;
        let unsafe_units = rd()?;
        let size_bound = rd()?;
        let len = rd()? as usize;
        let body = bytes.get(pos..pos + len).ok_or(ArtefactError::Truncated)?;
        let term = Term::decode(body).map_err(ArtefactError::Term)?;
        Ok(Artefact { header: Header { scheme, source_hash: Hash32(h), family, units, unsafe_units, size_bound }, term })
    }
}
