//! BLAKE2b-256, in crate, no C, no dependency (spec §7.5). RFC 7693.

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct Hash32(pub [u8; 32]);

impl Hash32 {
    pub fn hex(&self) -> String {
        let mut s = String::with_capacity(64);
        for b in self.0 {
            s.push_str(&format!("{b:02x}"));
        }
        s
    }
    pub fn short(&self) -> String {
        self.hex()[..8].to_string()
    }
}

impl std::fmt::Debug for Hash32 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "#{}", self.short())
    }
}

pub trait Digest32: Send + Sync + 'static {
    fn digest(bytes: &[u8]) -> Hash32;
}

pub struct Blake2b256;

impl Digest32 for Blake2b256 {
    fn digest(bytes: &[u8]) -> Hash32 {
        blake2b_256(bytes)
    }
}

const IV: [u64; 8] = [
    0x6a09e667f3bcc908,
    0xbb67ae8584caa73b,
    0x3c6ef372fe94f82b,
    0xa54ff53a5f1d36f1,
    0x510e527fade682d1,
    0x9b05688c2b3e6c1f,
    0x1f83d9abfb41bd6b,
    0x5be0cd19137e2179,
];

const SIGMA: [[usize; 16]; 12] = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    [14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3],
    [11, 8, 12, 0, 5, 2, 15, 13, 10, 14, 3, 6, 7, 1, 9, 4],
    [7, 9, 3, 1, 13, 12, 11, 14, 2, 6, 5, 10, 4, 0, 15, 8],
    [9, 0, 5, 7, 2, 4, 10, 15, 14, 1, 11, 12, 6, 8, 3, 13],
    [2, 12, 6, 10, 0, 11, 8, 3, 4, 13, 7, 5, 15, 14, 1, 9],
    [12, 5, 1, 15, 14, 13, 4, 10, 0, 7, 6, 3, 9, 2, 8, 11],
    [13, 11, 7, 14, 12, 1, 3, 9, 5, 0, 15, 4, 8, 6, 2, 10],
    [6, 15, 14, 9, 11, 3, 0, 8, 12, 2, 13, 7, 1, 4, 10, 5],
    [10, 2, 8, 4, 7, 6, 1, 5, 15, 11, 9, 14, 3, 12, 13, 0],
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    [14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3],
];

#[inline]
fn g(v: &mut [u64; 16], a: usize, b: usize, c: usize, d: usize, x: u64, y: u64) {
    v[a] = v[a].wrapping_add(v[b]).wrapping_add(x);
    v[d] = (v[d] ^ v[a]).rotate_right(32);
    v[c] = v[c].wrapping_add(v[d]);
    v[b] = (v[b] ^ v[c]).rotate_right(24);
    v[a] = v[a].wrapping_add(v[b]).wrapping_add(y);
    v[d] = (v[d] ^ v[a]).rotate_right(16);
    v[c] = v[c].wrapping_add(v[d]);
    v[b] = (v[b] ^ v[c]).rotate_right(63);
}

fn compress(h: &mut [u64; 8], block: &[u8; 128], t: u128, last: bool) {
    let mut m = [0u64; 16];
    for i in 0..16 {
        let mut w = [0u8; 8];
        w.copy_from_slice(&block[i * 8..i * 8 + 8]);
        m[i] = u64::from_le_bytes(w);
    }
    let mut v = [0u64; 16];
    v[..8].copy_from_slice(h);
    v[8..].copy_from_slice(&IV);
    v[12] ^= t as u64;
    v[13] ^= (t >> 64) as u64;
    if last {
        v[14] = !v[14];
    }
    for s in SIGMA.iter() {
        g(&mut v, 0, 4, 8, 12, m[s[0]], m[s[1]]);
        g(&mut v, 1, 5, 9, 13, m[s[2]], m[s[3]]);
        g(&mut v, 2, 6, 10, 14, m[s[4]], m[s[5]]);
        g(&mut v, 3, 7, 11, 15, m[s[6]], m[s[7]]);
        g(&mut v, 0, 5, 10, 15, m[s[8]], m[s[9]]);
        g(&mut v, 1, 6, 11, 12, m[s[10]], m[s[11]]);
        g(&mut v, 2, 7, 8, 13, m[s[12]], m[s[13]]);
        g(&mut v, 3, 4, 9, 14, m[s[14]], m[s[15]]);
    }
    for i in 0..8 {
        h[i] ^= v[i] ^ v[i + 8];
    }
}

pub fn blake2b_256(input: &[u8]) -> Hash32 {
    let mut h = IV;
    h[0] ^= 0x0101_0000 ^ 32u64; // no key, 32-byte digest
    let mut t: u128 = 0;
    let n = input.len();
    let full = if n == 0 { 0 } else { (n - 1) / 128 };
    for i in 0..full {
        let mut block = [0u8; 128];
        block.copy_from_slice(&input[i * 128..i * 128 + 128]);
        t += 128;
        compress(&mut h, &block, t, false);
    }
    let mut block = [0u8; 128];
    let rest = &input[full * 128..];
    block[..rest.len()].copy_from_slice(rest);
    t += rest.len() as u128;
    compress(&mut h, &block, t, true);
    let mut out = [0u8; 32];
    for i in 0..4 {
        out[i * 8..i * 8 + 8].copy_from_slice(&h[i].to_le_bytes());
    }
    Hash32(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rfc_vectors() {
        // BLAKE2b-256 of the empty string and of "abc".
        assert_eq!(
            blake2b_256(b"").hex(),
            "0e5751c026e543b2e8ab2eb06099daa1d1e5df47778f7787faab45cdf12fe3a8"
        );
        assert_eq!(
            blake2b_256(b"abc").hex(),
            "bddd813c634239723171ef3fee98579b94964e3bb1cb3e427262c8c068d52319"
        );
    }

    #[test]
    fn long_input_spans_blocks() {
        let input = vec![0x61u8; 300];
        let h = blake2b_256(&input);
        // Stability, not a published vector: the same bytes must hash the same way
        // on every target, which is what the determinism requirement needs.
        assert_eq!(h, blake2b_256(&input));
        assert_ne!(h, blake2b_256(&input[..299]));
    }
}
