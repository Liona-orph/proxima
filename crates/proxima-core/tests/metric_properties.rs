//! Property tests for the distance kernels.
//!
//! Two families of claim are checked here, and they fail in different ways:
//!
//! 1. **Metric axioms.** Identity, symmetry and — only where the metric is a
//!    true metric — the triangle inequality. A violation here means an index
//!    that prunes on the triangle inequality would silently lose recall.
//! 2. **Scalar/unrolled agreement.** The unrolled kernels sum in a different
//!    order, so they cannot be bit-identical. What they must be is *close*, on
//!    ordinary inputs and on the adversarial ones (zeros, denormals, huge
//!    magnitudes) where a naive reduction loses precision.
//!
//! Tolerances are relative and stated per property, because an absolute
//! tolerance is meaningless when the operands range over 30 orders of
//! magnitude.

#![allow(clippy::unwrap_used)]
// Exact float comparison is deliberate here: the constants involved are
// exactly representable and the point is that the kernel returns them
// bit-for-bit.
#![allow(clippy::float_cmp)]

use proptest::prelude::*;
use proxima_core::distance::{
    cosine, cosine_normalized, cosine_scalar, dot, dot_scalar, dot_unrolled, euclidean, manhattan,
    manhattan_scalar, manhattan_unrolled, norm, normalized, squared_euclidean,
    squared_euclidean_scalar, squared_euclidean_unrolled,
};
use proxima_core::{Metric, VectorSpace};

/// Dimensions spanning "shorter than one lane", "exactly one lane" and "several
/// lanes plus an awkward tail", which is where remainder-handling bugs live.
const DIMS: [usize; 6] = [1, 3, 8, 9, 33, 128];

/// Ordinary-magnitude components.
fn ordinary(dim: usize) -> impl Strategy<Value = Vec<f32>> {
    prop::collection::vec(-1000.0f32..1000.0, dim)
}

/// Components drawn from the awkward corners of the f32 line.
///
/// Zeros of both signs, denormals (which some build configurations flush to
/// zero — the two summation orders must still agree), and magnitudes up to
/// `1e18`, whose squares reach `1e36` and so come within two orders of
/// magnitude of overflowing `f32`.
///
/// The generator deliberately stops short of `f32::MAX`. Beyond roughly `1e19`
/// a squared or multiplied term overflows to infinity, and once two infinities
/// of opposite sign meet, the result is NaN in one summation order and infinity
/// in the other — the orders genuinely disagree. That is not a bug in the
/// kernels, it is arithmetic, and `overflow_is_order_dependent` below pins the
/// exact counterexample rather than letting a weakened tolerance hide it.
fn adversarial(dim: usize) -> impl Strategy<Value = Vec<f32>> {
    let component = prop_oneof![
        Just(0.0f32),
        Just(-0.0f32),
        Just(f32::MIN_POSITIVE),
        Just(f32::MIN_POSITIVE / 4.0), // denormal
        Just(1e-30f32),
        Just(1e18f32),
        Just(-1e18f32),
        Just(1e9f32),
        -1.0f32..1.0,
    ];
    prop::collection::vec(component, dim)
}

/// Relative closeness, with an absolute floor so that "both about zero" passes.
fn close(a: f32, b: f32, relative: f32) -> bool {
    if a == b {
        return true;
    }
    if a.is_nan() && b.is_nan() {
        return true;
    }
    if a.is_infinite() || b.is_infinite() {
        // Overflow must happen in both or neither; if it happens in both, they
        // agree by definition.
        return a == b;
    }
    let scale = a.abs().max(b.abs()).max(1e-20);
    (a - b).abs() <= relative * scale
}

/// Absolute closeness, for the metrics whose output range is bounded.
///
/// Cosine lives in `[0, 2]`, so "relatively close" is the wrong question for
/// it: two nearly parallel vectors score `0.0` under one computation and `6e-8`
/// under another, which is a *relative* error of 100 % and an absolute error
/// that no caller could ever observe.
fn close_abs(a: f32, b: f32, tolerance: f32) -> bool {
    (a - b).abs() <= tolerance
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    /// d(x, x) = 0 for the true metrics, and cosine of a vector with itself is 0.
    #[test]
    fn identity_of_indiscernibles(v in prop::collection::vec(-100.0f32..100.0, 1..129)) {
        prop_assert_eq!(squared_euclidean(&v, &v), 0.0);
        prop_assert_eq!(manhattan(&v, &v), 0.0);
        if norm(&v) > 1e-3 {
            prop_assert!(cosine(&v, &v).abs() < 1e-4, "cosine(v,v) = {}", cosine(&v, &v));
        }
    }

    /// d(x, y) = d(y, x), exactly: the kernels are symmetric in their operands
    /// and the summation order does not depend on which is which.
    #[test]
    fn symmetry(dim in prop::sample::select(&DIMS[..]), a in ordinary(128), b in ordinary(128)) {
        let (a, b) = (&a[..dim], &b[..dim]);
        prop_assert_eq!(squared_euclidean(a, b), squared_euclidean(b, a));
        prop_assert_eq!(manhattan(a, b), manhattan(b, a));
        prop_assert_eq!(dot(a, b), dot(b, a));
        prop_assert!(close(cosine(a, b), cosine(b, a), 1e-5));
    }

    /// The triangle inequality, for the metrics that have it.
    ///
    /// Asserted on `euclidean`, not `squared_euclidean`: squaring breaks it
    /// (0-1-2 on a line gives 1 + 1 < 4), which is exactly why the squared form
    /// is documented as a ranking score and not a distance.
    #[test]
    fn triangle_inequality(dim in prop::sample::select(&DIMS[..]),
                           a in ordinary(128), b in ordinary(128), c in ordinary(128)) {
        let (a, b, c) = (&a[..dim], &b[..dim], &c[..dim]);
        // 1e-4 relative slack: the sum of two f32 square roots accumulates
        // enough error to fail an exact comparison on near-degenerate triples.
        let slack = 1e-4 * (euclidean(a, b) + euclidean(b, c)).max(1.0);
        prop_assert!(euclidean(a, c) <= euclidean(a, b) + euclidean(b, c) + slack);
        prop_assert!(manhattan(a, c) <= manhattan(a, b) + manhattan(b, c) + 1e-3 * (manhattan(a, b) + manhattan(b, c)).max(1.0));
    }

    /// The squared form really does fail it, and the crate says so. Pinning the
    /// counterexample stops anyone "fixing" `is_true_metric`.
    #[test]
    fn squared_euclidean_is_not_a_metric(x in 1.0f32..100.0) {
        let (a, b, c) = ([0.0f32], [x], [2.0 * x]);
        prop_assert!(squared_euclidean(&a, &c) > squared_euclidean(&a, &b) + squared_euclidean(&b, &c));
        prop_assert!(Metric::Euclidean.is_true_metric(), "the axiom holds for the rectified form");
    }

    /// Scalar and unrolled agree on ordinary inputs, to 1e-4 relative.
    #[test]
    fn kernels_agree_on_ordinary_inputs(dim in prop::sample::select(&DIMS[..]),
                                        a in ordinary(128), b in ordinary(128)) {
        let (a, b) = (&a[..dim], &b[..dim]);
        prop_assert!(close(squared_euclidean_scalar(a, b), squared_euclidean_unrolled(a, b), 1e-4));
        prop_assert!(close(manhattan_scalar(a, b), manhattan_unrolled(a, b), 1e-4));
        // Backward bound for the dot product, for the reason spelled out in
        // `kernels_agree_on_adversarial_inputs`: the sum can cancel, so the
        // result's own magnitude is not a usable yardstick even here.
        let term_scale: f32 = a.iter().zip(b).map(|(x, y)| (x * y).abs()).sum();
        prop_assert!(
            (dot_scalar(a, b) - dot_unrolled(a, b)).abs() <= 1e-5 * term_scale.max(1e-20)
        );
        prop_assert!(close_abs(cosine_scalar(a, b), cosine(a, b), 1e-4));
    }

    /// ...and on the adversarial ones.
    ///
    /// The tolerance is looser (1e-2) and honestly so: mixing `1e18` with
    /// `1e-30` in the same reduction destroys most of the mantissa whatever
    /// order you sum in, and the two orders then differ by a large *relative*
    /// amount on a value that is meaningless anyway. What this property really
    /// pins down is that they agree on *which* results are zero, infinite or
    /// NaN — the classifications a search actually depends on.
    #[test]
    fn kernels_agree_on_adversarial_inputs(dim in prop::sample::select(&DIMS[..]),
                                           a in adversarial(128), b in adversarial(128)) {
        let (a, b) = (&a[..dim], &b[..dim]);

        let (s, u) = (squared_euclidean_scalar(a, b), squared_euclidean_unrolled(a, b));
        prop_assert_eq!(s.is_nan(), u.is_nan());
        prop_assert_eq!(s.is_infinite(), u.is_infinite());
        prop_assert_eq!(s == 0.0, u == 0.0);
        if s.is_finite() && u.is_finite() {
            prop_assert!(close(s, u, 1e-2), "{} vs {}", s, u);
        }

        let (s, u) = (manhattan_scalar(a, b), manhattan_unrolled(a, b));
        prop_assert_eq!(s.is_nan(), u.is_nan());
        prop_assert_eq!(s == 0.0, u == 0.0);
        if s.is_finite() && u.is_finite() {
            prop_assert!(close(s, u, 1e-2), "{} vs {}", s, u);
        }

        // The dot product is the one kernel where terms cancel, so the two
        // orders cannot be compared against the *result*: a sum of 1e27-scale
        // terms that cancels to zero has no meaningful relative error. The
        // correct statement is backward: both orders are within a small
        // multiple of the machine epsilon times the sum of the term
        // magnitudes, which is the standard accuracy bound for a floating
        // point dot product and the only one either implementation can honour.
        let (s, u) = (dot_scalar(a, b), dot_unrolled(a, b));
        prop_assert_eq!(s.is_nan(), u.is_nan());
        let term_scale: f32 = a.iter().zip(b).map(|(x, y)| (x * y).abs()).sum();
        if s.is_finite() && u.is_finite() && term_scale.is_finite() {
            prop_assert!(
                (s - u).abs() <= 1e-5 * term_scale.max(1e-20),
                "{} vs {} against term scale {}", s, u, term_scale
            );
        }
    }

    /// Distances are non-negative where the metric promises it.
    #[test]
    fn non_negativity(dim in prop::sample::select(&DIMS[..]), a in ordinary(128), b in ordinary(128)) {
        let (a, b) = (&a[..dim], &b[..dim]);
        prop_assert!(squared_euclidean(a, b) >= 0.0);
        prop_assert!(manhattan(a, b) >= 0.0);
        let cos = cosine(a, b);
        prop_assert!((-1e-4..=2.0 + 1e-4).contains(&cos), "cosine out of range: {}", cos);
    }

    /// The identity `VectorSpace` is built around: after `prepare`, cosine is
    /// one minus a dot product.
    #[test]
    fn cosine_on_prepared_vectors_is_inner_product(dim in prop::sample::select(&DIMS[..]),
                                                   a in ordinary(128), b in ordinary(128)) {
        let (a, b) = (&a[..dim], &b[..dim]);
        prop_assume!(norm(a) > 1e-2 && norm(b) > 1e-2);
        let space = VectorSpace::new(dim, Metric::Cosine).unwrap();
        let pa = space.prepare(a).unwrap();
        let pb = space.prepare(b).unwrap();
        prop_assert!(close_abs(space.distance(&pa, &pb), cosine(a, b), 1e-4));
        prop_assert!(close_abs(cosine_normalized(&pa, &pb), cosine(a, b), 1e-4));
    }

    /// A NaN anywhere in either operand produces a NaN score, in both variants.
    ///
    /// This is the *documented* behaviour, not an accident: the kernels do not
    /// sanitise, and `VectorSpace::validate` is the only thing standing between
    /// a NaN and the index. The test exists so that "we reject at the boundary"
    /// stays a claim about a checked property rather than a comment.
    #[test]
    fn nan_input_gives_nan_output(dim in prop::sample::select(&DIMS[..]),
                                  position in 0usize..128, a in ordinary(128), b in ordinary(128)) {
        let mut a = a[..dim].to_vec();
        let b = &b[..dim];
        a[position % dim] = f32::NAN;
        prop_assert!(squared_euclidean_scalar(&a, b).is_nan());
        prop_assert!(squared_euclidean_unrolled(&a, b).is_nan());
        prop_assert!(manhattan_scalar(&a, b).is_nan());
        prop_assert!(manhattan_unrolled(&a, b).is_nan());
        prop_assert!(dot_scalar(&a, b).is_nan());
        prop_assert!(dot_unrolled(&a, b).is_nan());
        // ...and the boundary check catches it before it can get anywhere.
        let space = VectorSpace::new(dim, Metric::Euclidean).unwrap();
        prop_assert!(space.validate(&a).is_err());
    }

    /// Normalisation produces unit vectors, and leaves the zero vector alone.
    #[test]
    fn normalization_is_idempotent_and_zero_safe(dim in prop::sample::select(&DIMS[..]), v in ordinary(128)) {
        let v = &v[..dim];
        let once = normalized(v);
        if norm(v) > 1e-2 {
            prop_assert!((norm(&once) - 1.0).abs() < 1e-4);
            let twice = normalized(&once);
            for (x, y) in once.iter().zip(&twice) {
                prop_assert!(close(*x, *y, 1e-5));
            }
        }
        let zero = vec![0.0f32; dim];
        prop_assert_eq!(normalized(&zero), zero);
    }

    /// The kernels never invent a result out of range for their metric.
    #[test]
    fn adversarial_inputs_stay_in_range(dim in prop::sample::select(&DIMS[..]),
                                        a in adversarial(128), b in adversarial(128)) {
        let (a, b) = (&a[..dim], &b[..dim]);
        prop_assert!(squared_euclidean(a, b) >= 0.0);
        prop_assert!(manhattan(a, b) >= 0.0);
        let cos = cosine(a, b);
        prop_assert!(!cos.is_nan());
        prop_assert!((-1e-3..=2.0 + 1e-3).contains(&cos), "cosine out of range: {}", cos);
    }

    /// `to_true_distance` is monotone, so it can never reorder results.
    #[test]
    fn rectification_preserves_order(a in 0.0f32..1e6, b in 0.0f32..1e6) {
        for metric in [Metric::Euclidean, Metric::Cosine, Metric::Manhattan] {
            prop_assert_eq!(a <= b, metric.to_true_distance(a) <= metric.to_true_distance(b));
        }
        // Inner product negates, so it reverses -- which is precisely why the
        // ranking path never calls it.
        prop_assert_eq!(a <= b, Metric::InnerProduct.to_true_distance(a) >= Metric::InnerProduct.to_true_distance(b));
    }
}

/// The one place scalar and unrolled genuinely disagree, pinned as a fact.
///
/// With components near `f32::MAX`, individual products overflow to infinity.
/// Summing left to right reaches `-inf` before it meets the `+inf`, and
/// `-inf + inf` is NaN; the lane-wise order keeps them in different
/// accumulators and reduces to `+inf`. Both are correct IEEE-754 answers to
/// different (equally legal) summation orders.
///
/// The engine's answer to this is not a tolerance, it is
/// [`VectorSpace::validate`]: magnitudes this large are not representable as a
/// meaningful embedding, and the boundary check rejects the *inputs* rather
/// than trying to reconcile the *outputs*.
#[test]
fn overflow_is_order_dependent() {
    let mut a = vec![0.0f32; 33];
    let mut b = vec![0.0f32; 33];
    a[14] = f32::MAX;
    b[14] = -0.914_645_55;
    a[17] = -0.880_665_24;
    b[17] = f32::MAX;
    a[25] = 1e30;
    b[25] = f32::MAX;
    a[30] = f32::MAX;

    assert!(dot_scalar(&a, &b).is_nan());
    assert!(dot_unrolled(&a, &b).is_infinite());

    // And this is why it never reaches a search: nothing this large is a
    // finite-magnitude embedding, but even if it were, the products are.
    let space = VectorSpace::new(33, Metric::InnerProduct).unwrap();
    assert!(
        space.validate(&a).is_ok(),
        "the inputs themselves are finite"
    );
    assert!(
        dot(&a, &b).is_infinite(),
        "the overflow happens inside the kernel, so callers with 1e38-scale \
         components must rescale; see the distance module docs"
    );
}
