//! The byte-level encoding shared by the write-ahead log and the snapshot.
//!
//! # Why a hand-written codec and not `serde`
//!
//! Two reasons, and the second is the one that decided it.
//!
//! The first is the workspace dependency policy: `serde` plus a binary format
//! crate is a large amount of compiled code and a derive macro on every type in
//! `proxima-core`, in exchange for saving the four hundred lines below.
//!
//! The second is that this is an **on-disk format**, and an on-disk format is a
//! compatibility promise. With derives, the layout is a consequence of the
//! field order of a struct that somebody will one day reorder for readability,
//! and nothing will fail until a user's snapshot does not load. Written out by
//! hand, the format is a thing you can read, version and test — and
//! [`crate::snapshot`] does test it, against a byte string checked into the
//! repository.
//!
//! # The format
//!
//! Little-endian throughout, because every machine this is likely to run on is
//! little-endian and byte-swapping on the write path of a log is a cost paid on
//! every record for a portability nobody has asked for. A big-endian reader
//! would produce garbage rather than an error, so the snapshot header records a
//! byte-order mark; see [`crate::snapshot`].
//!
//! Lengths are `u32`. That caps a metadata string at 4 GiB, which is not a
//! limit anyone will meet, and it keeps the per-field overhead at four bytes
//! rather than the eight a `u64` would cost on a corpus with a dozen short
//! metadata fields per vector.

use std::collections::BTreeMap;

use proxima_core::{Metadata, MetadataValue, Metric, ProximaError, Result, VectorSpace};

/// Appends primitives to a byte buffer.
///
/// Infallible: a `Vec<u8>` cannot fail to grow, and threading a `Result`
/// through every field of a serialiser that cannot fail buys nothing.
#[derive(Debug, Default)]
pub struct Writer {
    buffer: Vec<u8>,
}

impl Writer {
    /// An empty writer.
    pub fn new() -> Self {
        Writer::default()
    }

    /// An empty writer with `capacity` bytes reserved.
    pub fn with_capacity(capacity: usize) -> Self {
        Writer {
            buffer: Vec::with_capacity(capacity),
        }
    }

    /// The bytes written so far.
    pub fn as_slice(&self) -> &[u8] {
        &self.buffer
    }

    /// Consume the writer, yielding the buffer.
    pub fn into_bytes(self) -> Vec<u8> {
        self.buffer
    }

    /// Bytes written so far.
    pub fn len(&self) -> usize {
        self.buffer.len()
    }

    /// `true` if nothing has been written.
    pub fn is_empty(&self) -> bool {
        self.buffer.is_empty()
    }

    /// Append one byte.
    pub fn u8(&mut self, value: u8) -> &mut Self {
        self.buffer.push(value);
        self
    }

    /// Append a little-endian `u32`.
    pub fn u32(&mut self, value: u32) -> &mut Self {
        self.buffer.extend_from_slice(&value.to_le_bytes());
        self
    }

    /// Append a little-endian `u64`.
    pub fn u64(&mut self, value: u64) -> &mut Self {
        self.buffer.extend_from_slice(&value.to_le_bytes());
        self
    }

    /// Append a little-endian `f32`.
    ///
    /// `to_le_bytes` on the IEEE-754 bit pattern, so a stored vector round-trips
    /// bit-for-bit. That is not a nicety: the whole point of a snapshot test is
    /// that reloading reproduces the *same* search results, and a float that
    /// came back one ulp different would change a tie-break.
    pub fn f32(&mut self, value: f32) -> &mut Self {
        self.buffer.extend_from_slice(&value.to_le_bytes());
        self
    }

    /// Append a little-endian `f64`.
    pub fn f64(&mut self, value: f64) -> &mut Self {
        self.buffer.extend_from_slice(&value.to_le_bytes());
        self
    }

    /// Append a `usize` as a `u32`, saturating.
    ///
    /// Saturation rather than truncation: every caller writes a length that a
    /// `u32` comfortably holds, and if one ever did not, a value of
    /// `u32::MAX` fails the reader's bounds check loudly instead of wrapping to
    /// a small number that reads a plausible prefix.
    pub fn len_prefix(&mut self, value: usize) -> &mut Self {
        self.u32(u32::try_from(value).unwrap_or(u32::MAX))
    }

    /// Append bytes verbatim, with no length prefix.
    ///
    /// Only for the fixed-width parts of a file header, where the length is a
    /// constant of the format rather than data.
    pub fn raw(&mut self, value: &[u8]) -> &mut Self {
        self.buffer.extend_from_slice(value);
        self
    }

    /// Append a length-prefixed byte string.
    pub fn bytes(&mut self, value: &[u8]) -> &mut Self {
        self.len_prefix(value.len());
        self.buffer.extend_from_slice(value);
        self
    }

    /// Append a length-prefixed UTF-8 string.
    pub fn str(&mut self, value: &str) -> &mut Self {
        self.bytes(value.as_bytes())
    }

    /// Append a length-prefixed `f32` array.
    pub fn f32_slice(&mut self, values: &[f32]) -> &mut Self {
        self.len_prefix(values.len());
        for value in values {
            self.buffer.extend_from_slice(&value.to_le_bytes());
        }
        self
    }

    /// Append a length-prefixed `u32` array.
    pub fn u32_slice(&mut self, values: &[u32]) -> &mut Self {
        self.len_prefix(values.len());
        for value in values {
            self.buffer.extend_from_slice(&value.to_le_bytes());
        }
        self
    }

    /// Append a length-prefixed `u8` array without the `bytes` indirection.
    pub fn u8_slice(&mut self, values: &[u8]) -> &mut Self {
        self.bytes(values)
    }

    /// Append a [`VectorSpace`] as dimension plus a metric tag.
    pub fn space(&mut self, space: &VectorSpace) -> &mut Self {
        self.u32(space.dim() as u32);
        self.u8(metric_tag(space.metric()))
    }

    /// Append [`Metadata`], **with the keys sorted**.
    ///
    /// Sorted because `Metadata` is backed by a `HashMap` and its iteration
    /// order varies between processes. Without the sort, two snapshots of the
    /// same index would differ byte-for-byte, which would make the format
    /// impossible to test against a golden vector and would defeat any attempt
    /// to deduplicate or diff snapshots. The sort costs `O(f log f)` on a
    /// handful of fields per vector and is paid only when writing.
    pub fn metadata(&mut self, metadata: &Metadata) -> &mut Self {
        let ordered: BTreeMap<&str, &MetadataValue> = metadata.iter().collect();
        self.len_prefix(ordered.len());
        for (key, value) in ordered {
            self.str(key);
            match value {
                MetadataValue::Text(text) => {
                    self.u8(TAG_TEXT).str(text);
                }
                MetadataValue::Int(number) => {
                    self.u8(TAG_INT).u64(*number as u64);
                }
                MetadataValue::Float(number) => {
                    self.u8(TAG_FLOAT).f64(*number);
                }
                MetadataValue::Bool(flag) => {
                    self.u8(TAG_BOOL).u8(u8::from(*flag));
                }
            }
        }
        self
    }
}

/// Reads primitives back out of a byte slice.
///
/// Every method returns [`ProximaError::Corruption`] on a short read rather
/// than panicking, because the input is a file that may have been truncated by
/// a power failure — a malformed buffer is an expected condition here, not a
/// bug.
#[derive(Debug)]
pub struct Reader<'a> {
    bytes: &'a [u8],
    position: usize,
    /// Named in every error, so a corrupt file says which file.
    structure: &'static str,
}

impl<'a> Reader<'a> {
    /// A reader over `bytes`, reporting failures against `structure`.
    pub fn new(bytes: &'a [u8], structure: &'static str) -> Self {
        Reader {
            bytes,
            position: 0,
            structure,
        }
    }

    /// Bytes not yet consumed.
    pub fn remaining(&self) -> usize {
        self.bytes.len() - self.position
    }

    /// `true` if everything has been consumed.
    pub fn is_empty(&self) -> bool {
        self.remaining() == 0
    }

    /// Fail unless the buffer is fully consumed.
    ///
    /// Called at the end of every top-level decode. Trailing bytes mean the
    /// writer and the reader disagree about the format, which is exactly the
    /// class of bug a version number alone does not catch.
    pub fn expect_end(&self) -> Result<()> {
        if self.remaining() != 0 {
            return Err(ProximaError::corrupt(
                self.structure,
                format!("{} trailing bytes after the final field", self.remaining()),
            ));
        }
        Ok(())
    }

    fn take(&mut self, count: usize) -> Result<&'a [u8]> {
        if self.remaining() < count {
            return Err(ProximaError::corrupt(
                self.structure,
                format!(
                    "truncated at byte {}: wanted {count} more, {} remain",
                    self.position,
                    self.remaining()
                ),
            ));
        }
        let slice = &self.bytes[self.position..self.position + count];
        self.position += count;
        Ok(slice)
    }

    /// Read one byte.
    pub fn u8(&mut self) -> Result<u8> {
        Ok(self.take(1)?[0])
    }

    /// Read a little-endian `u32`.
    pub fn u32(&mut self) -> Result<u32> {
        let bytes = self.take(4)?;
        Ok(u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
    }

    /// Read a little-endian `u64`.
    pub fn u64(&mut self) -> Result<u64> {
        let bytes = self.take(8)?;
        let mut array = [0u8; 8];
        array.copy_from_slice(bytes);
        Ok(u64::from_le_bytes(array))
    }

    /// Read a little-endian `f32`.
    pub fn f32(&mut self) -> Result<f32> {
        Ok(f32::from_bits(self.u32()?))
    }

    /// Read a little-endian `f64`.
    pub fn f64(&mut self) -> Result<f64> {
        Ok(f64::from_bits(self.u64()?))
    }

    /// Read a length prefix, rejecting one that cannot be satisfied.
    ///
    /// The bound is the single most important check in the whole reader: a
    /// corrupt four-byte length is otherwise a four-gigabyte allocation, and a
    /// process that is `OOM`-killed while opening a damaged file cannot report
    /// what was wrong with it.
    pub fn len_prefix(&mut self, element_bytes: usize) -> Result<usize> {
        let length = self.u32()? as usize;
        let needed = length.saturating_mul(element_bytes.max(1));
        if needed > self.remaining() {
            return Err(ProximaError::corrupt(
                self.structure,
                format!(
                    "length prefix at byte {} claims {length} elements ({needed} bytes) with only {} left",
                    self.position - 4,
                    self.remaining()
                ),
            ));
        }
        Ok(length)
    }

    /// Read a length-prefixed byte string.
    pub fn bytes(&mut self) -> Result<&'a [u8]> {
        let length = self.len_prefix(1)?;
        self.take(length)
    }

    /// Read a length-prefixed UTF-8 string.
    pub fn str(&mut self) -> Result<&'a str> {
        let bytes = self.bytes()?;
        std::str::from_utf8(bytes).map_err(|err| {
            ProximaError::corrupt(self.structure, format!("field is not UTF-8: {err}"))
        })
    }

    /// Read a length-prefixed `f32` array.
    pub fn f32_vec(&mut self) -> Result<Vec<f32>> {
        let length = self.len_prefix(4)?;
        let bytes = self.take(length * 4)?;
        Ok(bytes
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect())
    }

    /// Read a length-prefixed `u32` array.
    pub fn u32_vec(&mut self) -> Result<Vec<u32>> {
        let length = self.len_prefix(4)?;
        let bytes = self.take(length * 4)?;
        Ok(bytes
            .chunks_exact(4)
            .map(|chunk| u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect())
    }

    /// Read a length-prefixed `u8` array.
    pub fn u8_vec(&mut self) -> Result<Vec<u8>> {
        Ok(self.bytes()?.to_vec())
    }

    /// Read a [`VectorSpace`].
    pub fn space(&mut self) -> Result<VectorSpace> {
        let dim = self.u32()? as usize;
        let metric = metric_from_tag(self.u8()?, self.structure)?;
        VectorSpace::new(dim, metric)
    }

    /// Read [`Metadata`].
    pub fn metadata(&mut self) -> Result<Metadata> {
        // Four bytes per field is the floor for a key length alone, so this
        // bounds the count without over-promising.
        let count = self.len_prefix(4)?;
        let mut metadata = Metadata::new();
        for _ in 0..count {
            let key = self.str()?.to_owned();
            let tag = self.u8()?;
            let value = match tag {
                TAG_TEXT => MetadataValue::Text(self.str()?.to_owned()),
                TAG_INT => MetadataValue::Int(self.u64()? as i64),
                TAG_FLOAT => MetadataValue::Float(self.f64()?),
                TAG_BOOL => MetadataValue::Bool(self.u8()? != 0),
                other => {
                    return Err(ProximaError::corrupt(
                        self.structure,
                        format!("metadata value tag {other} is not one of 1..=4"),
                    ))
                }
            };
            metadata.insert(key, value);
        }
        Ok(metadata)
    }
}

const TAG_TEXT: u8 = 1;
const TAG_INT: u8 = 2;
const TAG_FLOAT: u8 = 3;
const TAG_BOOL: u8 = 4;

/// The on-disk tag for a metric.
///
/// Explicit numbers rather than `as u8` on the enum: `Metric`'s declaration
/// order is a source-code detail and reordering it must not silently reinterpret
/// every snapshot on disk.
pub fn metric_tag(metric: Metric) -> u8 {
    match metric {
        Metric::Euclidean => 1,
        Metric::Cosine => 2,
        Metric::InnerProduct => 3,
        Metric::Manhattan => 4,
    }
}

/// The inverse of [`metric_tag`].
pub fn metric_from_tag(tag: u8, structure: &'static str) -> Result<Metric> {
    match tag {
        1 => Ok(Metric::Euclidean),
        2 => Ok(Metric::Cosine),
        3 => Ok(Metric::InnerProduct),
        4 => Ok(Metric::Manhattan),
        other => Err(ProximaError::corrupt(
            structure,
            format!("metric tag {other} is not one of 1..=4"),
        )),
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    #![allow(clippy::float_cmp)]

    use super::*;

    #[test]
    fn primitives_round_trip() {
        let mut writer = Writer::new();
        writer
            .u8(0xAB)
            .u32(0xDEAD_BEEF)
            .u64(0x0123_4567_89AB_CDEF)
            .f32(-1.5)
            .f64(std::f64::consts::PI)
            .str("héllo")
            .f32_slice(&[1.0, 2.0, 3.0])
            .u32_slice(&[7, 8])
            .u8_slice(&[9, 10, 11]);
        let bytes = writer.into_bytes();

        let mut reader = Reader::new(&bytes, "test");
        assert_eq!(reader.u8().unwrap(), 0xAB);
        assert_eq!(reader.u32().unwrap(), 0xDEAD_BEEF);
        assert_eq!(reader.u64().unwrap(), 0x0123_4567_89AB_CDEF);
        assert_eq!(reader.f32().unwrap(), -1.5);
        assert_eq!(reader.f64().unwrap(), std::f64::consts::PI);
        assert_eq!(reader.str().unwrap(), "héllo");
        assert_eq!(reader.f32_vec().unwrap(), vec![1.0, 2.0, 3.0]);
        assert_eq!(reader.u32_vec().unwrap(), vec![7, 8]);
        assert_eq!(reader.u8_vec().unwrap(), vec![9, 10, 11]);
        reader.expect_end().unwrap();
    }

    #[test]
    fn metadata_round_trips_and_is_key_ordered() {
        let metadata = Metadata::new()
            .with("zebra", "stripes")
            .with("apple", 3_i64)
            .with("score", 0.5_f64)
            .with("live", true);

        // Written many times, the bytes must be identical: the HashMap's
        // iteration order is not, and the sort is what hides that.
        let encode = |m: &Metadata| {
            let mut w = Writer::new();
            w.metadata(m);
            w.into_bytes()
        };
        let first = encode(&metadata);
        for _ in 0..16 {
            assert_eq!(encode(&metadata.clone()), first);
        }

        let mut reader = Reader::new(&first, "test");
        assert_eq!(reader.metadata().unwrap(), metadata);
        reader.expect_end().unwrap();
    }

    #[test]
    fn empty_metadata_costs_four_bytes() {
        let mut writer = Writer::new();
        writer.metadata(&Metadata::new());
        assert_eq!(writer.len(), 4);
    }

    #[test]
    fn space_round_trips_every_metric() {
        for metric in Metric::ALL {
            let space = VectorSpace::new(11, metric).unwrap();
            let mut writer = Writer::new();
            writer.space(&space);
            let bytes = writer.into_bytes();
            let mut reader = Reader::new(&bytes, "test");
            assert_eq!(reader.space().unwrap(), space);
        }
    }

    #[test]
    fn a_truncated_buffer_is_an_error_not_a_panic() {
        let mut writer = Writer::new();
        writer.str("a reasonably long string");
        let bytes = writer.into_bytes();
        for keep in 0..bytes.len() {
            let mut reader = Reader::new(&bytes[..keep], "test");
            assert!(reader.str().is_err(), "kept {keep} bytes");
        }
    }

    #[test]
    fn an_absurd_length_prefix_is_rejected_before_allocating() {
        // The whole point of `len_prefix`: four bytes of garbage must not
        // become a four-gigabyte allocation.
        let bytes = [0xFFu8, 0xFF, 0xFF, 0xFF, 0x00];
        let mut reader = Reader::new(&bytes, "test");
        let err = reader.f32_vec().unwrap_err();
        assert!(err.to_string().contains("4294967295"), "{err}");
    }

    #[test]
    fn unknown_tags_are_named_in_the_error() {
        assert!(metric_from_tag(9, "test")
            .unwrap_err()
            .to_string()
            .contains('9'));

        let bytes = {
            let mut writer = Writer::new();
            writer.u32(1).str("k").u8(99);
            writer.into_bytes()
        };
        let err = Reader::new(&bytes, "test").metadata().unwrap_err();
        assert!(err.to_string().contains("99"), "{err}");
    }

    #[test]
    fn trailing_bytes_are_reported() {
        let bytes = [1u8, 2, 3];
        let mut reader = Reader::new(&bytes, "test");
        assert_eq!(reader.u8().unwrap(), 1);
        assert!(reader.expect_end().is_err());
    }
}
