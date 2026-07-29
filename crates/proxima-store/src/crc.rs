//! CRC-32C (Castagnoli), the integrity check on every record and every
//! snapshot.
//!
//! # Why CRC-32C and not CRC-32, or a hash
//!
//! The job is narrow: detect a *torn* write — a record the kernel only partly
//! flushed before the machine died — and detect bit rot on a page that came
//! back wrong from the disk. It is not to detect tampering, so a cryptographic
//! hash would be four times the code and ten times the cost for a property
//! nothing here relies on.
//!
//! Castagnoli's polynomial rather than the classic zlib one for two reasons.
//! It has better minimum Hamming distance at the record sizes this log writes
//! (a few hundred bytes to a few kilobytes), and it is the polynomial x86-64
//! implements in hardware as `crc32` and aarch64 as `crc32c`, so a future
//! `unsafe`-and-intrinsics version of this file is a drop-in replacement that
//! does not change a single byte on disk. The table-driven version here runs at
//! roughly 1 GiB/s, which is far above what an fsync-per-record log can push.
//!
//! # What it does not catch
//!
//! A 32-bit check has a `2^-32` chance of accepting a corrupt record, so a log
//! that is torn once a day will accept a bad record roughly once every eleven
//! million years. It catches every single-bit error, every burst up to 32 bits,
//! and — the case that actually happens — a record whose tail is simply
//! missing, because the length prefix and the payload then disagree.

/// The Castagnoli polynomial, bit-reversed for the table-driven form.
const POLYNOMIAL: u32 = 0x82F6_3B78;

/// One byte's worth of remainders, built once on first use.
///
/// A `const fn` and a `static`, not a `OnceLock`: this is consulted per byte of
/// every record written and read, and a lazy-initialisation branch on that path
/// is both a needless check and a needless atomic load. 1 KiB of `.rodata` is
/// the whole cost.
static TABLE: [u32; 256] = build_table();

const fn build_table() -> [u32; 256] {
    let mut table = [0u32; 256];
    let mut byte = 0usize;
    while byte < 256 {
        let mut remainder = byte as u32;
        let mut bit = 0;
        while bit < 8 {
            remainder = if remainder & 1 == 1 {
                (remainder >> 1) ^ POLYNOMIAL
            } else {
                remainder >> 1
            };
            bit += 1;
        }
        table[byte] = remainder;
        byte += 1;
    }
    table
}

/// CRC-32C of `bytes`.
///
/// ```
/// // The check value every CRC-32C implementation agrees on.
/// assert_eq!(proxima_store::crc32c(b"123456789"), 0xE306_9283);
/// assert_eq!(proxima_store::crc32c(b""), 0);
/// ```
#[must_use]
pub fn crc32c(bytes: &[u8]) -> u32 {
    crc32c_append(0, bytes)
}

/// Continue a CRC-32C over another chunk.
///
/// Exists so that a record's header and its payload can be checksummed without
/// first being concatenated into one buffer — the log writes them as two
/// slices and copying a 3 KiB vector to compute a checksum over it would be the
/// single largest allocation on the write path.
#[must_use]
pub fn crc32c_append(crc: u32, bytes: &[u8]) -> u32 {
    let mut remainder = !crc;
    for byte in bytes {
        let index = ((remainder ^ u32::from(*byte)) & 0xFF) as usize;
        remainder = TABLE[index] ^ (remainder >> 8);
    }
    !remainder
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use super::*;

    #[test]
    fn matches_the_published_check_value() {
        // Every CRC-32C implementation in the world agrees on this one; it is
        // the reason the constant is worth asserting rather than golden-testing
        // against our own output.
        assert_eq!(crc32c(b"123456789"), 0xE306_9283);
    }

    #[test]
    fn empty_input_is_zero() {
        assert_eq!(crc32c(&[]), 0);
    }

    #[test]
    fn chunking_does_not_change_the_answer() {
        let data: Vec<u8> = (0..=255u8).cycle().take(4097).collect();
        let whole = crc32c(&data);
        for split in [1usize, 17, 256, 4096] {
            let (head, tail) = data.split_at(split);
            assert_eq!(crc32c_append(crc32c(head), tail), whole, "split at {split}");
        }
    }

    #[test]
    fn a_single_flipped_bit_changes_it() {
        let data = vec![0x5Au8; 512];
        let clean = crc32c(&data);
        for bit in [0usize, 7, 100, 4095] {
            let mut corrupt = data.clone();
            corrupt[bit / 8] ^= 1 << (bit % 8);
            assert_ne!(crc32c(&corrupt), clean, "bit {bit}");
        }
    }

    #[test]
    fn truncation_changes_it() {
        // The failure this check actually exists for: a record whose tail never
        // reached the platter.
        let data: Vec<u8> = (0..300u32).map(|i| i as u8).collect();
        let full = crc32c(&data);
        for keep in [0usize, 1, 299] {
            assert_ne!(crc32c(&data[..keep]), full, "kept {keep}");
        }
    }
}
