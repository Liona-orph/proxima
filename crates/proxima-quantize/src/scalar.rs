//! Scalar quantization: one byte per dimension, via an affine map.
//!
//! The simplest useful compression there is. Each component is mapped linearly
//! onto `0..=255` and stored as a byte, so a 768-dimensional `f32` vector goes
//! from 3072 bytes to 768 — exactly 4x, always, with no codebook to train
//! beyond two floats per dimension.
//!
//! It is worth having alongside [`ProductQuantizer`](crate::ProductQuantizer)
//! because the trade-offs are genuinely different: SQ costs 4x compression for
//! a very small accuracy loss and a decode that is one multiply-add, whereas PQ
//! costs 32x-192x for a much larger loss and needs training. On a corpus that
//! fits in RAM at 4x but not at 1x, SQ is the whole answer.

use proxima_core::{ProximaError, Result, VectorSpace};

/// How the byte pattern is interpreted.
///
/// Both variants describe the *same* 256-point lattice; they differ only in
/// whether the stored byte is biased by 128. That sounds like a distinction
/// without a difference, and it is — until you write a SIMD kernel, at which
/// point the instruction you want (`vdot`/`vpdpbusd`-style widening dot
/// products) is defined on signed lanes on some targets and unsigned on
/// others, and re-biasing a loaded vector costs an extra operation per lane.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CodeType {
    /// Codes are `0..=255`, stored directly.
    U8,
    /// Codes are `-128..=127`, stored as the byte `code + 128`.
    I8,
}

impl CodeType {
    /// The byte that stores lattice level `level` (which is always `0..=255`).
    #[inline]
    const fn encode_level(self, level: u8) -> u8 {
        match self {
            CodeType::U8 => level,
            // Two's complement of `level - 128`, so that reading the byte as an
            // `i8` gives a value centred on zero.
            CodeType::I8 => level.wrapping_sub(128),
        }
    }

    /// The lattice level a stored byte represents. Inverse of
    /// [`CodeType::encode_level`].
    #[inline]
    const fn decode_level(self, byte: u8) -> u8 {
        match self {
            CodeType::U8 => byte,
            CodeType::I8 => byte.wrapping_add(128),
        }
    }
}

/// Whether the quantizer keeps one range for the whole vector or one per
/// dimension.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RangeMode {
    /// One `(min, max)` for every component of every vector.
    ///
    /// Two floats total. Correct when components are drawn from one
    /// distribution — which is true of most normalised embeddings — and badly
    /// wrong when one dimension has a range a hundred times wider than the
    /// others, because that one dimension then consumes the entire lattice and
    /// every other dimension collapses onto a handful of levels.
    Global,
    /// One `(min, max)` per dimension.
    ///
    /// `2 * dim` floats of overhead — 6 KiB for a 768-dimensional space, i.e.
    /// the size of two stored vectors, so it is free at any realistic corpus
    /// size — and it removes the failure above entirely. The default.
    PerDimension,
}

/// Affine `f32` to one-byte quantizer.
///
/// # Asymmetric distance
///
/// The point of [`ScalarQuantizer::asymmetric_squared_l2`] is that the *query*
/// stays in `f32`. Decoding a stored code to compare it with a full-precision
/// query loses only the stored vector's precision; quantizing the query too
/// (the "symmetric" computation) loses both, roughly doubling the error, and
/// buys nothing here because there is only one query and millions of stored
/// vectors. Symmetric distance is worth it only when it lets you precompute a
/// lookup table, which is the product quantizer's trick and not available for
/// a per-dimension affine map.
///
/// ```
/// use proxima_core::{Metric, VectorSpace};
/// use proxima_quantize::{CodeType, RangeMode, ScalarQuantizer};
///
/// let space = VectorSpace::new(4, Metric::Euclidean)?;
/// let training: Vec<f32> = vec![
///     0.0, 1.0, 2.0, 3.0,
///     1.0, 0.0, 3.0, 2.0,
///     0.5, 0.5, 2.5, 2.5,
/// ];
///
/// let sq = ScalarQuantizer::train(&space, &training, RangeMode::PerDimension, CodeType::U8)?;
/// assert_eq!(sq.compression_ratio(), 4.0);
///
/// let code = sq.encode(&[0.25, 0.75, 2.25, 2.75])?;
/// assert_eq!(code.len(), 4);
///
/// let decoded = sq.decode(&code)?;
/// for (want, got) in [0.25, 0.75, 2.25, 2.75].iter().zip(&decoded) {
///     assert!((want - got).abs() < 0.01, "{want} vs {got}");
/// }
/// # Ok::<(), proxima_core::ProximaError>(())
/// ```
#[derive(Debug, Clone)]
pub struct ScalarQuantizer {
    dim: usize,
    mode: RangeMode,
    code_type: CodeType,
    /// Lower bound per dimension. Length 1 under [`RangeMode::Global`].
    min: Vec<f32>,
    /// `(max - min) / 255` per dimension: the width of one lattice step.
    scale: Vec<f32>,
    /// `1 / scale`, precomputed because encoding divides once per component and
    /// a reciprocal multiply is several times cheaper than a divide.
    inverse_scale: Vec<f32>,
}

/// Number of representable levels. Fixed at 256 because the code is one byte;
/// a `levels` parameter would only be useful with bit-packing, which is a
/// different data structure.
const LEVELS: usize = 256;

impl ScalarQuantizer {
    /// Fit the ranges to a training set.
    ///
    /// `training` is a flat `n * dim` array. Ranges are the observed min and
    /// max, not a percentile: clipping outliers would improve the average error
    /// and introduce an unbounded worst case, and an index that is *usually*
    /// accurate is harder to reason about than one that is uniformly slightly
    /// lossy.
    ///
    /// A dimension whose training values are all equal gets a zero scale, and
    /// reconstructs to that constant exactly; no division by zero occurs.
    pub fn train(
        space: &VectorSpace,
        training: &[f32],
        mode: RangeMode,
        code_type: CodeType,
    ) -> Result<Self> {
        let dim = space.dim();
        if training.is_empty() || training.len() % dim != 0 {
            return Err(ProximaError::config(
                "training",
                training.len(),
                "must be a non-empty multiple of the space dimension",
            ));
        }

        let groups = match mode {
            RangeMode::Global => 1,
            RangeMode::PerDimension => dim,
        };
        let mut min = vec![f32::INFINITY; groups];
        let mut max = vec![f32::NEG_INFINITY; groups];

        for vector in training.chunks_exact(dim) {
            for (component, value) in vector.iter().enumerate() {
                if !value.is_finite() {
                    return Err(ProximaError::NonFinite {
                        index: component,
                        value: *value,
                    });
                }
                let group = if groups == 1 { 0 } else { component };
                min[group] = min[group].min(*value);
                max[group] = max[group].max(*value);
            }
        }

        let scale: Vec<f32> = min
            .iter()
            .zip(&max)
            .map(|(lo, hi)| (hi - lo) / (LEVELS - 1) as f32)
            .collect();
        let inverse_scale = scale
            .iter()
            .map(|s| if *s > 0.0 { 1.0 / s } else { 0.0 })
            .collect();

        Ok(ScalarQuantizer {
            dim,
            mode,
            code_type,
            min,
            scale,
            inverse_scale,
        })
    }

    /// The dimension this quantizer was trained for.
    pub const fn dim(&self) -> usize {
        self.dim
    }

    /// Bytes one code occupies: one per dimension.
    pub const fn code_len(&self) -> usize {
        self.dim
    }

    /// How ranges are shared across dimensions.
    pub const fn range_mode(&self) -> RangeMode {
        self.mode
    }

    /// How stored bytes are interpreted.
    pub const fn code_type(&self) -> CodeType {
        self.code_type
    }

    /// Ratio of uncompressed to compressed size, ignoring the range table.
    ///
    /// Always exactly 4: `f32` in, `u8` out. Stated as a method anyway so that
    /// benchmark output and the PQ equivalent read the same way.
    pub fn compression_ratio(&self) -> f32 {
        std::mem::size_of::<f32>() as f32
    }

    /// Bytes of trained state, i.e. the range table.
    pub fn table_bytes(&self) -> usize {
        (self.min.len() + self.scale.len() + self.inverse_scale.len()) * std::mem::size_of::<f32>()
    }

    /// Quantize one vector.
    ///
    /// Values outside the trained range are clamped, not rejected: a corpus
    /// drifts, and refusing to index a vector because one component is 1 %
    /// beyond the training maximum would be a worse failure than representing
    /// it as the extreme level.
    pub fn encode(&self, vector: &[f32]) -> Result<Vec<u8>> {
        let mut out = vec![0u8; self.dim];
        self.encode_into(vector, &mut out)?;
        Ok(out)
    }

    /// [`ScalarQuantizer::encode`] into a caller-owned buffer.
    ///
    /// Exists because a bulk encode of a million vectors should not perform a
    /// million allocations.
    pub fn encode_into(&self, vector: &[f32], out: &mut [u8]) -> Result<()> {
        if vector.len() != self.dim {
            return Err(ProximaError::DimensionMismatch {
                expected: self.dim,
                actual: vector.len(),
            });
        }
        if out.len() != self.dim {
            return Err(ProximaError::config(
                "out",
                out.len(),
                "output buffer must be code_len() bytes",
            ));
        }
        for (component, value) in vector.iter().enumerate() {
            let group = self.group(component);
            let level = ((value - self.min[group]) * self.inverse_scale[group])
                .round()
                .clamp(0.0, (LEVELS - 1) as f32) as u8;
            out[component] = self.code_type.encode_level(level);
        }
        Ok(())
    }

    /// Reconstruct an approximation of the original vector.
    pub fn decode(&self, code: &[u8]) -> Result<Vec<f32>> {
        if code.len() != self.dim {
            return Err(ProximaError::DimensionMismatch {
                expected: self.dim,
                actual: code.len(),
            });
        }
        Ok((0..self.dim)
            .map(|component| self.reconstruct(component, code[component]))
            .collect())
    }

    /// Squared L2 between a full-precision query and a stored code.
    ///
    /// The asymmetric computation: the query is never quantized. One
    /// reconstruction and one squared difference per dimension, so it is about
    /// twice the cost of an `f32` squared-L2 kernel — but it reads a quarter of
    /// the bytes, which is the trade that matters once the corpus no longer
    /// fits in cache.
    ///
    /// # Panics
    ///
    /// If `code` is not [`ScalarQuantizer::code_len`] bytes. This is the inner
    /// loop of a scan; validating on every call would cost more than the
    /// arithmetic. Callers obtain codes from [`ScalarQuantizer::encode`], which
    /// does validate.
    #[inline]
    pub fn asymmetric_squared_l2(&self, query: &[f32], code: &[u8]) -> f32 {
        let mut sum = 0.0f32;
        for (component, (q, c)) in query.iter().zip(code).enumerate() {
            let d = q - self.reconstruct(component, *c);
            sum += d * d;
        }
        sum
    }

    /// Negated dot product between a full-precision query and a stored code.
    ///
    /// The inner-product and (on prepared vectors) cosine counterpart to
    /// [`ScalarQuantizer::asymmetric_squared_l2`]. Negated for the same reason
    /// [`proxima_core::distance::inner_product`] is: everything ranks min-first.
    #[inline]
    pub fn asymmetric_inner_product(&self, query: &[f32], code: &[u8]) -> f32 {
        let mut sum = 0.0f32;
        for (component, (q, c)) in query.iter().zip(code).enumerate() {
            sum += q * self.reconstruct(component, *c);
        }
        -sum
    }

    /// Value a stored byte represents in a given dimension.
    #[inline]
    fn reconstruct(&self, component: usize, byte: u8) -> f32 {
        let group = self.group(component);
        let level = self.code_type.decode_level(byte);
        self.min[group] + f32::from(level) * self.scale[group]
    }

    /// Which range entry a dimension uses.
    #[inline]
    const fn group(&self, component: usize) -> usize {
        match self.mode {
            RangeMode::Global => 0,
            RangeMode::PerDimension => component,
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    // Exact float comparison is deliberate here: every constant below is
    // exactly representable and the point is that the kernel returns it
    // bit-for-bit.
    #![allow(clippy::float_cmp)]

    use super::*;
    use proxima_core::Metric;

    fn space(dim: usize) -> VectorSpace {
        VectorSpace::new(dim, Metric::Euclidean).unwrap()
    }

    /// Two dimensions with wildly different ranges: the case `Global` gets
    /// wrong and `PerDimension` gets right.
    fn skewed_training() -> Vec<f32> {
        let mut out = Vec::new();
        for i in 0..256 {
            let t = i as f32 / 255.0;
            out.extend_from_slice(&[t, 1000.0 * t]);
        }
        out
    }

    #[test]
    fn round_trip_error_is_bounded_by_half_a_step() {
        let sq = ScalarQuantizer::train(
            &space(2),
            &skewed_training(),
            RangeMode::PerDimension,
            CodeType::U8,
        )
        .unwrap();
        for original in skewed_training().chunks_exact(2) {
            let decoded = sq.decode(&sq.encode(original).unwrap()).unwrap();
            for (component, (want, got)) in original.iter().zip(&decoded).enumerate() {
                let step = if component == 0 {
                    1.0 / 255.0
                } else {
                    1000.0 / 255.0
                };
                assert!(
                    (want - got).abs() <= step / 2.0 + 1e-5,
                    "component {component}: {want} vs {got}"
                );
            }
        }
    }

    #[test]
    fn global_range_collapses_a_narrow_dimension() {
        // The documented failure mode of `Global`, pinned so nobody "simplifies"
        // `PerDimension` away.
        let training = skewed_training();
        let global =
            ScalarQuantizer::train(&space(2), &training, RangeMode::Global, CodeType::U8).unwrap();
        let per_dim =
            ScalarQuantizer::train(&space(2), &training, RangeMode::PerDimension, CodeType::U8)
                .unwrap();

        let probe = [0.5f32, 500.0];
        let global_error = (global.decode(&global.encode(&probe).unwrap()).unwrap()[0] - 0.5).abs();
        let per_dim_error =
            (per_dim.decode(&per_dim.encode(&probe).unwrap()).unwrap()[0] - 0.5).abs();
        assert!(
            global_error > 100.0 * per_dim_error,
            "global {global_error} vs per-dimension {per_dim_error}"
        );
    }

    #[test]
    fn both_code_types_describe_the_same_lattice() {
        let training = skewed_training();
        let u8q =
            ScalarQuantizer::train(&space(2), &training, RangeMode::PerDimension, CodeType::U8)
                .unwrap();
        let i8q =
            ScalarQuantizer::train(&space(2), &training, RangeMode::PerDimension, CodeType::I8)
                .unwrap();
        for original in training.chunks_exact(2) {
            assert_eq!(
                u8q.decode(&u8q.encode(original).unwrap()).unwrap(),
                i8q.decode(&i8q.encode(original).unwrap()).unwrap()
            );
        }
    }

    #[test]
    fn out_of_range_values_clamp_rather_than_fail() {
        let sq = ScalarQuantizer::train(
            &space(1),
            &[0.0, 1.0],
            RangeMode::PerDimension,
            CodeType::U8,
        )
        .unwrap();
        assert_eq!(sq.decode(&sq.encode(&[-5.0]).unwrap()).unwrap(), vec![0.0]);
        assert_eq!(sq.decode(&sq.encode(&[9.0]).unwrap()).unwrap(), vec![1.0]);
    }

    #[test]
    fn constant_dimension_reconstructs_exactly() {
        let sq = ScalarQuantizer::train(
            &space(2),
            &[3.0, 0.0, 3.0, 1.0],
            RangeMode::PerDimension,
            CodeType::U8,
        )
        .unwrap();
        assert_eq!(sq.decode(&sq.encode(&[3.0, 0.5]).unwrap()).unwrap()[0], 3.0);
    }

    #[test]
    fn asymmetric_distance_tracks_the_exact_one() {
        let training = skewed_training();
        let sq =
            ScalarQuantizer::train(&space(2), &training, RangeMode::PerDimension, CodeType::U8)
                .unwrap();
        let query = [0.3f32, 700.0];
        for original in training.chunks_exact(2) {
            let code = sq.encode(original).unwrap();
            let approximate = sq.asymmetric_squared_l2(&query, &code);
            let exact = proxima_core::distance::squared_euclidean(&query, original);
            // The bound is loose because the error is dominated by the wide
            // dimension's step; what matters is that it is a bound at all.
            assert!(
                (approximate - exact).abs() <= 0.02 * exact.max(1.0),
                "{approximate} vs {exact}"
            );
        }
    }

    #[test]
    fn training_input_is_validated() {
        assert!(matches!(
            ScalarQuantizer::train(&space(3), &[], RangeMode::Global, CodeType::U8),
            Err(ProximaError::Configuration { .. })
        ));
        assert!(matches!(
            ScalarQuantizer::train(&space(3), &[1.0, 2.0], RangeMode::Global, CodeType::U8),
            Err(ProximaError::Configuration { .. })
        ));
        assert!(matches!(
            ScalarQuantizer::train(&space(2), &[1.0, f32::NAN], RangeMode::Global, CodeType::U8),
            Err(ProximaError::NonFinite { index: 1, .. })
        ));
    }

    #[test]
    fn encode_rejects_the_wrong_dimension() {
        let sq = ScalarQuantizer::train(&space(2), &[0.0, 1.0], RangeMode::Global, CodeType::U8)
            .unwrap();
        assert!(matches!(
            sq.encode(&[1.0]),
            Err(ProximaError::DimensionMismatch {
                expected: 2,
                actual: 1
            })
        ));
    }
}
