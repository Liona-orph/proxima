//! [`VectorSpace`]: dimension plus metric, and the one place normalisation is
//! decided.
//!
//! Every index, every quantizer and every recall measurement is parameterised
//! by one of these. It is small and `Copy`-free but cheap to clone, and it is
//! the type that makes the cosine trap hard to fall into.

use std::borrow::Cow;

use crate::distance::{self, Metric};
use crate::error::{ProximaError, Result};

/// The configuration a set of vectors is interpreted under.
///
/// # The cosine rule, enforced by the API
///
/// Cosine similarity of unit vectors is their dot product. Every serious vector
/// engine exploits this, and every engine that exposes it as a *convention*
/// eventually serves wrong results to someone who normalised the corpus and
/// forgot the query, or vice versa.
///
/// `VectorSpace` makes it structural instead:
///
/// * [`VectorSpace::prepare`] is the only supported way to hand a raw vector to
///   an index or a quantizer, and it normalises when
///   [`Metric::requires_normalization`] says to. It borrows when no work is
///   needed, so the non-cosine path does not allocate.
/// * [`VectorSpace::distance`] then runs [`distance::cosine_normalized`] — the
///   two-norm-free kernel — and is documented as requiring prepared operands.
///
/// The failure mode this removes is silent: an unnormalised query against a
/// normalised corpus still returns *plausible* neighbours, just the wrong ones,
/// and no test that does not measure recall will notice.
///
/// ```
/// use proxima_core::{Metric, VectorSpace};
///
/// let space = VectorSpace::new(3, Metric::Cosine)?;
///
/// // `prepare` normalises because the metric asks it to...
/// let a = space.prepare(&[3.0, 0.0, 4.0])?;
/// assert!((a[0] - 0.6).abs() < 1e-6);
///
/// // ...and the prepared kernel then agrees with the general one.
/// let b = space.prepare(&[0.0, 1.0, 0.0])?;
/// assert!((space.distance(&a, &b) - Metric::Cosine.distance(&[3.0, 0.0, 4.0], &[0.0, 1.0, 0.0])).abs() < 1e-6);
///
/// // Euclidean spaces borrow instead of copying.
/// let euclid = VectorSpace::new(3, Metric::Euclidean)?;
/// assert!(matches!(euclid.prepare(&[1.0, 2.0, 3.0])?, std::borrow::Cow::Borrowed(_)));
/// # Ok::<(), proxima_core::ProximaError>(())
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VectorSpace {
    dim: usize,
    metric: Metric,
}

/// Refusing zero-dimensional spaces is not pedantry: a zero-length vector makes
/// every distance zero, so every index degenerates into "return the first k".
const MIN_DIM: usize = 1;

/// An upper bound on dimension, so that `slot * dim` cannot overflow `usize` on
/// a 32-bit target and so that an obviously wrong config (a byte count passed
/// where a dimension was meant) fails at construction rather than at the first
/// allocation.
const MAX_DIM: usize = 1 << 20;

impl VectorSpace {
    /// Build a space, validating the dimension.
    pub fn new(dim: usize, metric: Metric) -> Result<Self> {
        if dim < MIN_DIM {
            return Err(ProximaError::config(
                "dim",
                dim,
                "must be at least 1; a 0-dimensional space makes all distances equal",
            ));
        }
        if dim > MAX_DIM {
            return Err(ProximaError::config(
                "dim",
                dim,
                "exceeds the 1048576 dimension ceiling",
            ));
        }
        Ok(VectorSpace { dim, metric })
    }

    /// The dimension every vector in this space must have.
    pub const fn dim(&self) -> usize {
        self.dim
    }

    /// The metric.
    pub const fn metric(&self) -> Metric {
        self.metric
    }

    /// Check length and finiteness.
    ///
    /// O(dim), and called once per insert and once per query — not per
    /// candidate. Rejecting NaN here is what lets the kernels skip checking; see
    /// the [`crate::distance`] module docs.
    pub fn validate(&self, vector: &[f32]) -> Result<()> {
        if vector.len() != self.dim {
            return Err(ProximaError::DimensionMismatch {
                expected: self.dim,
                actual: vector.len(),
            });
        }
        for (index, value) in vector.iter().enumerate() {
            if !value.is_finite() {
                return Err(ProximaError::NonFinite {
                    index,
                    value: *value,
                });
            }
        }
        Ok(())
    }

    /// Validate, and normalise if the metric needs it.
    ///
    /// Returns [`Cow::Borrowed`] when no transformation applies, so the
    /// Euclidean, Manhattan and inner-product paths are allocation-free. The
    /// cosine path allocates one `Vec<f32>` of `dim` floats per call; an index
    /// that inserts millions of vectors pays that once each, which is far
    /// cheaper than the two extra norm accumulators it buys back on every one
    /// of the hundreds of distance evaluations per query.
    pub fn prepare<'a>(&self, vector: &'a [f32]) -> Result<Cow<'a, [f32]>> {
        self.validate(vector)?;
        if !self.metric.requires_normalization() {
            return Ok(Cow::Borrowed(vector));
        }
        let mut owned = vector.to_vec();
        distance::normalize_in_place(&mut owned);
        Ok(Cow::Owned(owned))
    }

    /// Ranking score between two **prepared** vectors.
    ///
    /// This is the hot function of the entire engine: one call per graph edge
    /// examined. It assumes both operands came from [`VectorSpace::prepare`]
    /// and are therefore the right length and, for cosine, unit-length. Passing
    /// raw vectors under [`Metric::Cosine`] is not an error and returns a wrong
    /// answer — use [`Metric::distance`] if you cannot guarantee preparation.
    #[inline]
    pub fn distance(&self, a: &[f32], b: &[f32]) -> f32 {
        match self.metric {
            Metric::Euclidean => distance::squared_euclidean(a, b),
            // The whole point of `prepare`: no norms in the inner loop.
            Metric::Cosine => distance::cosine_normalized(a, b),
            Metric::InnerProduct => distance::inner_product(a, b),
            Metric::Manhattan => distance::manhattan(a, b),
        }
    }

    /// Bytes one vector occupies in a flat store.
    pub const fn vector_bytes(&self) -> usize {
        self.dim * std::mem::size_of::<f32>()
    }
}

impl std::fmt::Display for VectorSpace {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}d/{}", self.dim, self.metric)
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

    #[test]
    fn rejects_degenerate_dimensions() {
        assert!(matches!(
            VectorSpace::new(0, Metric::Euclidean),
            Err(ProximaError::Configuration {
                parameter: "dim",
                ..
            })
        ));
        assert!(VectorSpace::new(MAX_DIM + 1, Metric::Euclidean).is_err());
        assert!(VectorSpace::new(1, Metric::Euclidean).is_ok());
    }

    #[test]
    fn validate_names_both_dimensions() {
        let space = VectorSpace::new(4, Metric::Euclidean).unwrap();
        let err = space.validate(&[1.0, 2.0]).unwrap_err();
        assert_eq!(
            err,
            ProximaError::DimensionMismatch {
                expected: 4,
                actual: 2
            }
        );
    }

    #[test]
    fn validate_rejects_non_finite_and_says_where() {
        let space = VectorSpace::new(3, Metric::Euclidean).unwrap();
        let err = space.validate(&[1.0, f32::INFINITY, 3.0]).unwrap_err();
        assert!(matches!(err, ProximaError::NonFinite { index: 1, .. }));
        let err = space.validate(&[1.0, 2.0, f32::NAN]).unwrap_err();
        assert!(matches!(err, ProximaError::NonFinite { index: 2, .. }));
    }

    #[test]
    fn non_cosine_prepare_borrows() {
        for metric in [Metric::Euclidean, Metric::Manhattan, Metric::InnerProduct] {
            let space = VectorSpace::new(2, metric).unwrap();
            assert!(matches!(
                space.prepare(&[3.0, 4.0]).unwrap(),
                Cow::Borrowed(_)
            ));
        }
    }

    #[test]
    fn prepared_cosine_matches_the_general_kernel() {
        // The identity `VectorSpace` exists to enforce, asserted rather than
        // asserted-in-a-comment.
        let space = VectorSpace::new(5, Metric::Cosine).unwrap();
        let raw_a = [1.5, -2.0, 0.25, 8.0, -0.5];
        let raw_b = [-3.0, 1.0, 4.5, 0.125, 2.0];
        let a = space.prepare(&raw_a).unwrap();
        let b = space.prepare(&raw_b).unwrap();
        let fast = space.distance(&a, &b);
        let general = Metric::Cosine.distance(&raw_a, &raw_b);
        assert!((fast - general).abs() < 1e-6, "{fast} vs {general}");
    }

    #[test]
    fn display_is_loggable() {
        let space = VectorSpace::new(768, Metric::Cosine).unwrap();
        assert_eq!(space.to_string(), "768d/cosine");
    }
}
