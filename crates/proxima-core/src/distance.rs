//! Distance kernels, and the [`Metric`] that selects between them.
//!
//! # Everything here is a *ranking score*, not necessarily a distance
//!
//! A search only ever needs to order candidates. Every function in this module
//! returns a value that is **monotonically increasing in dissimilarity**, and
//! nothing more is promised. Two consequences that surprise people:
//!
//! * [`squared_euclidean`] is the Euclidean kernel. `sqrt` is monotone, so it
//!   cannot change an ordering; computing it would add one square root per
//!   candidate to a loop that does a few hundred multiply-adds, for a number
//!   nobody looks at. [`euclidean`] exists for when you do want to look, and
//!   for the triangle-inequality property test, which the squared form fails.
//! * [`inner_product`] returns the *negated* dot product, because larger dot
//!   products mean more similar and every ranked structure in this crate is
//!   min-first. Use [`Metric::to_true_distance`] to recover the presentable
//!   number.
//!
//! # Scalar and unrolled variants
//!
//! Each kernel appears twice, `*_scalar` and `*_unrolled`, and the public name
//! dispatches to the unrolled one. The scalar version is not dead code: it is
//! the reference implementation that the property tests check the unrolled one
//! against, and it is the honest baseline in `benches/distance.rs`.
//!
//! What the unrolling actually buys is not "fewer instructions" — it is
//! **breaking the loop-carried dependency on the accumulator**. IEEE-754
//! addition is not associative, so LLVM is not allowed to reassociate
//! `sum += x` into independent partial sums on its own; the naive loop is a
//! chain of dependent `fadd`s, each waiting on the previous one's ~3 cycle
//! latency, and no amount of `-O3` will vectorise it. Writing eight independent
//! accumulators states the reassociation explicitly, which is a *different*
//! (equally valid, differently rounded) summation order that the compiler can
//! then trivially map onto two 4-lane NEON `fmla`s per iteration on aarch64, or
//! one 8-lane AVX operation on x86-64.
//!
//! The cost is that the two variants do not produce bit-identical results. They
//! agree to a relative tolerance, and the property tests assert that tolerance
//! rather than equality.
//!
//! Do not take the paragraph above on faith — `benches/distance.rs` measures it.
//! On the benchmark machine (aarch64, rustc 1.83, `lto = "fat"`), at dimension
//! 768: squared Euclidean 349.9 ns to 115.9 ns (**3.0x**), dot 328.3 ns to
//! 45.1 ns (**7.3x**), Manhattan 323.3 ns to 47.4 ns (**6.8x**), cosine
//! 449.8 ns to 76.7 ns (**5.9x**). Squared Euclidean gains least because its
//! inner loop has three operations to dot product's two, so it starts out closer
//! to issue-limited than dependency-limited.
//!
//! # NaN and infinity
//!
//! The kernels **propagate** non-finite values: a NaN component anywhere in
//! either operand produces a NaN score. That is the correct arithmetic answer
//! and it is a terrible search result, because NaN compares false against
//! everything and would make a vector either invisible or omnipresent depending
//! on the comparison's direction.
//!
//! Proxima's policy is therefore to reject non-finite data **at the boundary**:
//! [`crate::VectorSpace::validate`] returns [`crate::ProximaError::NonFinite`]
//! and no such vector is ever stored. Should a NaN score reach a ranked
//! structure anyway, [`crate::SearchResult`] orders with `f32::total_cmp`,
//! under which NaN is larger than every finite value — so a NaN-scored
//! candidate loses every comparison and is dropped, rather than poisoning the
//! heap. Infinities need no special handling: they order correctly.

/// Lane count of the unrolled kernels.
///
/// Eight because it is two NEON `float32x4_t` registers or one AVX `__m256`,
/// and because eight independent accumulator chains cover the ~3-4 cycle
/// latency of a floating-point add on every core this is likely to run on.
/// Sixteen was measured and is not better: it spills on aarch64 for the
/// three-accumulator cosine kernel.
pub const LANES: usize = 8;

/// Which notion of "near" an index uses.
///
/// ```
/// use proxima_core::Metric;
///
/// let a = [3.0, 4.0];
/// let b = [0.0, 0.0];
///
/// // Euclidean ranks on the *squared* distance: 3^2 + 4^2.
/// assert_eq!(Metric::Euclidean.distance(&a, &b), 25.0);
/// // ...and can present the familiar one on request.
/// assert_eq!(Metric::Euclidean.to_true_distance(25.0), 5.0);
///
/// assert_eq!(Metric::Manhattan.distance(&a, &b), 7.0);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Metric {
    /// Squared L2. The default for embeddings that are not unit-normalised.
    ///
    /// A true metric once the square root is taken; see [`euclidean`].
    Euclidean,
    /// Angular distance, `1 - cos(a, b)`, in `[0, 2]`.
    ///
    /// The standalone [`Metric::distance`] computes both norms on every call,
    /// which is roughly twice the work of the inner-product kernel. This is
    /// avoidable and [`crate::VectorSpace`] avoids it — see that type's docs.
    ///
    /// Not a true metric: `1 - cos` violates the triangle inequality (the
    /// *angle* itself does not, but the engine never needs `acos`).
    Cosine,
    /// Negated dot product. Larger dot product, smaller score.
    ///
    /// Not a metric at all — a vector is not closest to itself under it, which
    /// is exactly why it is useful for maximum-inner-product search over
    /// vectors with meaningful magnitudes. Note that HNSW's proofs assume a
    /// metric space, so recall under `InnerProduct` is empirically good but not
    /// underwritten by anything; the recall harness is how you find out.
    InnerProduct,
    /// L1 / taxicab distance.
    ///
    /// A true metric. Cheaper than Euclidean (an absolute value instead of a
    /// multiply) and more robust to a single wildly wrong dimension, which is
    /// why it sometimes wins on sparse or count-like features.
    Manhattan,
}

impl Metric {
    /// Ranking score between two raw vectors.
    ///
    /// Self-contained and always correct, including for [`Metric::Cosine`],
    /// where it normalises on the fly. In an index's inner loop use
    /// [`crate::VectorSpace::distance`] instead, which knows the vectors were
    /// normalised on the way in.
    ///
    /// # Panics
    ///
    /// Does not panic on length mismatch; the kernels iterate over the shorter
    /// operand. Length is validated once, at insert and at query, by
    /// [`crate::VectorSpace::validate`].
    #[inline]
    pub fn distance(self, a: &[f32], b: &[f32]) -> f32 {
        match self {
            Metric::Euclidean => squared_euclidean(a, b),
            Metric::Cosine => cosine(a, b),
            Metric::InnerProduct => inner_product(a, b),
            Metric::Manhattan => manhattan(a, b),
        }
    }

    /// `true` if the metric satisfies identity, symmetry and the triangle
    /// inequality once [`Metric::to_true_distance`] has been applied.
    ///
    /// Consulted by the property tests, and worth consulting before assuming
    /// any pruning argument that needs the triangle inequality holds.
    pub const fn is_true_metric(self) -> bool {
        matches!(self, Metric::Euclidean | Metric::Manhattan)
    }

    /// `true` if vectors must be unit-length for the fast kernel to be correct.
    ///
    /// Only [`Metric::Cosine`]. [`crate::VectorSpace`] uses this to decide
    /// whether to normalise on the way in.
    pub const fn requires_normalization(self) -> bool {
        matches!(self, Metric::Cosine)
    }

    /// Convert a ranking score into the number a human expects to see.
    ///
    /// Takes the square root for [`Metric::Euclidean`] and undoes the negation
    /// for [`Metric::InnerProduct`]; the identity elsewhere. Purely
    /// presentational — never call it inside a ranking loop, because it is
    /// monotone and therefore cannot change the answer.
    #[inline]
    pub fn to_true_distance(self, score: f32) -> f32 {
        match self {
            Metric::Euclidean => score.sqrt(),
            Metric::InnerProduct => -score,
            Metric::Cosine | Metric::Manhattan => score,
        }
    }

    /// Lower bound on any score this metric can produce.
    ///
    /// Used by the ranked structures as the "cannot possibly be beaten" value.
    /// `InnerProduct` is unbounded below, so it reports `f32::NEG_INFINITY`.
    pub const fn min_score(self) -> f32 {
        match self {
            Metric::Euclidean | Metric::Cosine | Metric::Manhattan => 0.0,
            Metric::InnerProduct => f32::NEG_INFINITY,
        }
    }

    /// Parse the name [`Display`](std::fmt::Display) produces.
    ///
    /// The exact inverse of `to_string`, and a round-trip test asserts it. It
    /// exists because a metric arrives as a string at every boundary the engine
    /// has — a JSON request body, a CLI flag, a snapshot header — and three
    /// hand-written `match`es would be three chances to accept `"l2"` in one
    /// place and reject it in another.
    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            "euclidean" => Some(Metric::Euclidean),
            "cosine" => Some(Metric::Cosine),
            "inner_product" => Some(Metric::InnerProduct),
            "manhattan" => Some(Metric::Manhattan),
            _ => None,
        }
    }

    /// Every metric, in declaration order.
    ///
    /// For exhaustive tests and for listing the legal values in an error
    /// message, so that adding a metric cannot leave either behind.
    pub const ALL: [Metric; 4] = [
        Metric::Euclidean,
        Metric::Cosine,
        Metric::InnerProduct,
        Metric::Manhattan,
    ];
}

impl std::fmt::Display for Metric {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let name = match self {
            Metric::Euclidean => "euclidean",
            Metric::Cosine => "cosine",
            Metric::InnerProduct => "inner_product",
            Metric::Manhattan => "manhattan",
        };
        f.write_str(name)
    }
}

// ---------------------------------------------------------------------------
// Squared Euclidean
// ---------------------------------------------------------------------------

/// Squared L2 distance. The Euclidean ranking kernel.
///
/// Dispatches to [`squared_euclidean_unrolled`].
#[inline]
pub fn squared_euclidean(a: &[f32], b: &[f32]) -> f32 {
    squared_euclidean_unrolled(a, b)
}

/// True L2 distance, `sqrt` included.
///
/// Not used for ranking. It exists because the triangle inequality is a
/// property of *this* function and not of [`squared_euclidean`], and the
/// property tests need something to assert it against.
#[inline]
pub fn euclidean(a: &[f32], b: &[f32]) -> f32 {
    squared_euclidean(a, b).sqrt()
}

/// Reference implementation of [`squared_euclidean`]: one accumulator, in order.
#[inline]
pub fn squared_euclidean_scalar(a: &[f32], b: &[f32]) -> f32 {
    let mut sum = 0.0f32;
    for (x, y) in a.iter().zip(b) {
        let d = x - y;
        sum += d * d;
    }
    sum
}

/// [`LANES`]-way unrolled [`squared_euclidean`].
///
/// The zip over `acc`, `x` and `y` is deliberate: all three have statically
/// known length [`LANES`] inside the loop, so the iterator form compiles to
/// straight-line SIMD with no bounds checks, where indexing `x[i]` would leave
/// LLVM to prove the length of a `chunks_exact` item.
#[inline]
pub fn squared_euclidean_unrolled(a: &[f32], b: &[f32]) -> f32 {
    let mut acc = [0.0f32; LANES];
    let mut a_chunks = a.chunks_exact(LANES);
    let mut b_chunks = b.chunks_exact(LANES);
    for (x, y) in a_chunks.by_ref().zip(b_chunks.by_ref()) {
        for ((acc, x), y) in acc.iter_mut().zip(x).zip(y) {
            let d = x - y;
            *acc += d * d;
        }
    }
    let mut tail = 0.0f32;
    for (x, y) in a_chunks.remainder().iter().zip(b_chunks.remainder()) {
        let d = x - y;
        tail += d * d;
    }
    reduce(&acc) + tail
}

// ---------------------------------------------------------------------------
// Inner product
// ---------------------------------------------------------------------------

/// Negated dot product. The maximum-inner-product ranking kernel.
#[inline]
pub fn inner_product(a: &[f32], b: &[f32]) -> f32 {
    -dot_unrolled(a, b)
}

/// Plain dot product, sign intact.
///
/// Separate from [`inner_product`] because cosine, the quantizers and the
/// normalisation helpers all want the un-negated value, and burying a `-` in a
/// shared helper is how sign errors happen.
#[inline]
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    dot_unrolled(a, b)
}

/// Reference implementation of [`dot`].
#[inline]
pub fn dot_scalar(a: &[f32], b: &[f32]) -> f32 {
    let mut sum = 0.0f32;
    for (x, y) in a.iter().zip(b) {
        sum += x * y;
    }
    sum
}

/// [`LANES`]-way unrolled [`dot`].
#[inline]
pub fn dot_unrolled(a: &[f32], b: &[f32]) -> f32 {
    let mut acc = [0.0f32; LANES];
    let mut a_chunks = a.chunks_exact(LANES);
    let mut b_chunks = b.chunks_exact(LANES);
    for (x, y) in a_chunks.by_ref().zip(b_chunks.by_ref()) {
        for ((acc, x), y) in acc.iter_mut().zip(x).zip(y) {
            *acc += x * y;
        }
    }
    let mut tail = 0.0f32;
    for (x, y) in a_chunks.remainder().iter().zip(b_chunks.remainder()) {
        tail += x * y;
    }
    reduce(&acc) + tail
}

// ---------------------------------------------------------------------------
// Manhattan
// ---------------------------------------------------------------------------

/// L1 distance.
#[inline]
pub fn manhattan(a: &[f32], b: &[f32]) -> f32 {
    manhattan_unrolled(a, b)
}

/// Reference implementation of [`manhattan`].
#[inline]
pub fn manhattan_scalar(a: &[f32], b: &[f32]) -> f32 {
    let mut sum = 0.0f32;
    for (x, y) in a.iter().zip(b) {
        sum += (x - y).abs();
    }
    sum
}

/// [`LANES`]-way unrolled [`manhattan`].
///
/// `f32::abs` is a sign-bit clear, which vectorises; the reason this kernel
/// still needs unrolling is the same accumulator chain as everywhere else.
#[inline]
pub fn manhattan_unrolled(a: &[f32], b: &[f32]) -> f32 {
    let mut acc = [0.0f32; LANES];
    let mut a_chunks = a.chunks_exact(LANES);
    let mut b_chunks = b.chunks_exact(LANES);
    for (x, y) in a_chunks.by_ref().zip(b_chunks.by_ref()) {
        for ((acc, x), y) in acc.iter_mut().zip(x).zip(y) {
            *acc += (x - y).abs();
        }
    }
    let mut tail = 0.0f32;
    for (x, y) in a_chunks.remainder().iter().zip(b_chunks.remainder()) {
        tail += (x - y).abs();
    }
    reduce(&acc) + tail
}

// ---------------------------------------------------------------------------
// Cosine
// ---------------------------------------------------------------------------

/// Cosine distance, `1 - cos(a, b)`, computing both norms.
///
/// Returns `1.0` when either operand is the zero vector: the angle is undefined
/// there, and `1.0` is the score of an orthogonal pair, which is the least
/// wrong finite answer and keeps the result out of the top of the heap. The
/// alternative — NaN — is the failure mode described in the module docs.
#[inline]
pub fn cosine(a: &[f32], b: &[f32]) -> f32 {
    cosine_unrolled(a, b)
}

/// Cosine distance for operands that are already unit length.
///
/// This is the identity the whole cosine story rests on: with `|a| = |b| = 1`,
/// `1 - cos(a, b)` is `1 - a.b`, so cosine search *is* inner-product search and
/// the two norm accumulators disappear. [`crate::VectorSpace`] normalises on
/// insert and on query specifically so that this is the kernel that runs.
///
/// Feeding it non-unit vectors is not an error and produces a wrong answer, so
/// it is not reachable from [`Metric::distance`]; the only caller is
/// [`crate::VectorSpace::distance`], which owns the normalisation.
#[inline]
pub fn cosine_normalized(a: &[f32], b: &[f32]) -> f32 {
    1.0 - dot_unrolled(a, b)
}

/// Reference implementation of [`cosine`].
#[inline]
pub fn cosine_scalar(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0.0f32;
    let mut norm_a = 0.0f32;
    let mut norm_b = 0.0f32;
    for (x, y) in a.iter().zip(b) {
        dot += x * y;
        norm_a += x * x;
        norm_b += y * y;
    }
    let denom = (norm_a * norm_b).sqrt();
    if denom <= 0.0 {
        return 1.0;
    }
    1.0 - dot / denom
}

/// [`LANES`]-way unrolled [`cosine`].
#[inline]
pub fn cosine_unrolled(a: &[f32], b: &[f32]) -> f32 {
    let (dot, norm_a, norm_b) = cosine_terms_unrolled(a, b);
    let denom = (norm_a * norm_b).sqrt();
    if denom <= 0.0 {
        return 1.0;
    }
    1.0 - dot / denom
}

/// The three sums cosine needs, in a single pass over both operands.
///
/// One pass rather than three calls because the operands are read once and the
/// three accumulator arrays live in registers; splitting it into `dot` plus two
/// `norm`s triples the loads for no benefit. Three arrays is also where
/// [`LANES`] stops being free: at 8 lanes this is 24 live accumulators, which
/// aarch64 holds in its 32 vector registers and would not hold at 16 lanes.
#[inline]
fn cosine_terms_unrolled(a: &[f32], b: &[f32]) -> (f32, f32, f32) {
    let mut acc_dot = [0.0f32; LANES];
    let mut acc_a = [0.0f32; LANES];
    let mut acc_b = [0.0f32; LANES];
    let mut a_chunks = a.chunks_exact(LANES);
    let mut b_chunks = b.chunks_exact(LANES);
    for (x, y) in a_chunks.by_ref().zip(b_chunks.by_ref()) {
        for ((((d, na), nb), x), y) in acc_dot
            .iter_mut()
            .zip(acc_a.iter_mut())
            .zip(acc_b.iter_mut())
            .zip(x)
            .zip(y)
        {
            *d += x * y;
            *na += x * x;
            *nb += y * y;
        }
    }
    let (mut d, mut na, mut nb) = (0.0f32, 0.0f32, 0.0f32);
    for (x, y) in a_chunks.remainder().iter().zip(b_chunks.remainder()) {
        d += x * y;
        na += x * x;
        nb += y * y;
    }
    (
        reduce(&acc_dot) + d,
        reduce(&acc_a) + na,
        reduce(&acc_b) + nb,
    )
}

// ---------------------------------------------------------------------------
// Norms and normalisation
// ---------------------------------------------------------------------------

/// Squared L2 norm.
#[inline]
pub fn norm_squared(v: &[f32]) -> f32 {
    dot_unrolled(v, v)
}

/// L2 norm.
#[inline]
pub fn norm(v: &[f32]) -> f32 {
    norm_squared(v).sqrt()
}

/// Scale `v` to unit length in place; returns the norm it had.
///
/// A zero (or denormal-underflowing) vector is left untouched and `0.0` is
/// returned, because there is no direction to preserve and dividing would
/// produce NaN. Callers that care must check the return value — which is why it
/// is returned rather than discarded.
///
/// ```
/// use proxima_core::distance::{normalize_in_place, norm};
///
/// let mut v = [3.0, 4.0];
/// assert_eq!(normalize_in_place(&mut v), 5.0);
/// assert!((norm(&v) - 1.0).abs() < 1e-6);
///
/// let mut zero = [0.0, 0.0];
/// assert_eq!(normalize_in_place(&mut zero), 0.0);
/// assert_eq!(zero, [0.0, 0.0]);
/// ```
#[inline]
pub fn normalize_in_place(v: &mut [f32]) -> f32 {
    let n = norm(v);
    if n > 0.0 && n.is_finite() {
        let inv = 1.0 / n;
        for x in v.iter_mut() {
            *x *= inv;
        }
    }
    n
}

/// Allocating counterpart to [`normalize_in_place`].
#[inline]
pub fn normalized(v: &[f32]) -> Vec<f32> {
    let mut out = v.to_vec();
    normalize_in_place(&mut out);
    out
}

/// Sum the lane accumulators pairwise.
///
/// Pairwise rather than sequential so that the eight partial sums are combined
/// in a shallow tree; with `LANES = 8` that is three dependent adds instead of
/// seven, and it also gives the unrolled kernels marginally better error
/// behaviour than the scalar reference they are compared against.
#[inline]
fn reduce(acc: &[f32; LANES]) -> f32 {
    let a = (acc[0] + acc[1]) + (acc[2] + acc[3]);
    let b = (acc[4] + acc[5]) + (acc[6] + acc[7]);
    a + b
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
    fn known_values() {
        let a = [1.0, 2.0, 3.0];
        let b = [4.0, 6.0, 3.0];
        assert_eq!(squared_euclidean(&a, &b), 9.0 + 16.0);
        assert_eq!(euclidean(&a, &b), 5.0);
        assert_eq!(manhattan(&a, &b), 3.0 + 4.0);
        assert_eq!(dot(&a, &b), 4.0 + 12.0 + 9.0);
        assert_eq!(inner_product(&a, &b), -25.0);
    }

    #[test]
    fn cosine_of_orthogonal_and_parallel() {
        assert!((cosine(&[1.0, 0.0], &[0.0, 5.0]) - 1.0).abs() < 1e-6);
        assert!(cosine(&[1.0, 1.0], &[3.0, 3.0]).abs() < 1e-6);
        assert!((cosine(&[1.0, 0.0], &[-2.0, 0.0]) - 2.0).abs() < 1e-6);
    }

    #[test]
    fn cosine_on_zero_vector_is_finite() {
        // The documented fallback: undefined angle scores as orthogonal rather
        // than as NaN, so the vector loses instead of corrupting the heap.
        assert_eq!(cosine(&[0.0, 0.0], &[1.0, 1.0]), 1.0);
        assert_eq!(cosine_scalar(&[0.0, 0.0], &[1.0, 1.0]), 1.0);
    }

    #[test]
    fn cosine_on_unit_vectors_is_inner_product() {
        let a = normalized(&[0.3, -1.7, 2.5, 0.01, 4.0, -3.0, 1.0, 2.0, 9.0]);
        let b = normalized(&[1.0, 2.0, -0.5, 3.3, 0.0, 1.0, -1.0, 0.5, 0.25]);
        let general = cosine(&a, &b);
        let fast = cosine_normalized(&a, &b);
        assert!(
            (general - fast).abs() < 1e-5,
            "cosine {general} vs unit-vector fast path {fast}"
        );
    }

    #[test]
    fn tail_shorter_than_a_lane_is_not_dropped() {
        // The remainder loop is the easiest thing in this module to get wrong,
        // so every length from 0 to 3 full lanes is checked against the scalar
        // reference.
        for dim in 0..3 * LANES + 5 {
            let a: Vec<f32> = (0..dim).map(|i| i as f32 * 0.5 - 3.0).collect();
            let b: Vec<f32> = (0..dim).map(|i| 7.0 - i as f32 * 0.25).collect();
            let scalar = squared_euclidean_scalar(&a, &b);
            let unrolled = squared_euclidean_unrolled(&a, &b);
            assert!(
                (scalar - unrolled).abs() <= 1e-3 * scalar.abs().max(1.0),
                "dim {dim}: {scalar} vs {unrolled}"
            );
        }
    }

    #[test]
    fn nan_propagates_rather_than_being_silently_dropped() {
        // Documented policy: the kernels do not sanitise. Rejection happens at
        // the VectorSpace boundary, and this test pins the kernel half of it.
        let a = [1.0, f32::NAN, 3.0];
        let b = [0.0, 0.0, 0.0];
        assert!(squared_euclidean(&a, &b).is_nan());
        assert!(manhattan(&a, &b).is_nan());
        assert!(dot(&a, &b).is_nan());
    }

    #[test]
    fn metric_classification() {
        assert!(Metric::Euclidean.is_true_metric());
        assert!(Metric::Manhattan.is_true_metric());
        assert!(!Metric::Cosine.is_true_metric());
        assert!(!Metric::InnerProduct.is_true_metric());
        assert!(Metric::Cosine.requires_normalization());
        assert!(!Metric::Euclidean.requires_normalization());
    }
}
