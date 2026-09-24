"""End-to-end certification of Proposition 1 against a real sklearn RBF SVC.

The affine-identity tests in ``test_affine_audit.py`` check the *algebra*: that
a stationary kernel is translation invariant and that a linear head can absorb
an offset into its intercept. They do not fit anything. This module closes that
gap by training the actual estimator the manuscript's empirical centrepiece uses
-- ``sklearn.svm.SVC(kernel="rbf")`` -- and asserting the proposition's three
consequences directly:

1. translating every source training and validation vector by a fixed offset
   leaves the source-validation predictions unchanged,
2. the target decision function satisfies ``f_delta(x) = f_0(x - delta)``,
3. an unshifted target point can therefore change class, because ``x - delta``
   falls on the other side of a boundary.

**Exact mathematics, finite arithmetic.** Proposition 1 is an exact statement
about the fitting rule. Its numerical realisation is not bit-exact, and this
module does not pretend otherwise. ``sklearn.metrics.pairwise.rbf_kernel`` forms
squared distances by expanding them into norms and an inner product, and that
expansion is not translation invariant in floating point, so the two Gram
matrices agree to rounding rather than bitwise. The assertions below are stated
at a tolerance orders of magnitude tighter than any decision margin in play and
orders of magnitude looser than a bit-identity claim the mathematics does not
license. Discrete outcomes -- predicted labels, the selected support set -- are
asserted exactly, because those are not subject to rounding drift.

Nothing here reads the result ledger, the feature caches, or the corpora.
"""

from __future__ import annotations

import numpy as np
import pytest

from sklearn.metrics.pairwise import rbf_kernel
from sklearn.svm import SVC

# Observed worst-case disagreement on this fixture is ~4e-15 for the Gram
# matrices and ~4e-16 for decision values. 1e-12 leaves roughly three orders of
# headroom for a different BLAS or libsvm build while remaining far below the
# smallest decision margin these fixtures produce, so a genuine violation of
# the proposition could not hide underneath it.
NUMERICAL_ATOL = 1e-12

GAMMA = 0.25
FIT_PARAMS = dict(kernel="rbf", C=2.0, gamma=GAMMA, random_state=0, tol=1e-9)

# Three well-separated blobs. Separation is deliberate: a fit sitting on the
# edge of its own convergence tolerance would confound "the proposition fails"
# with "libsvm stopped somewhere else", and the proposition is a claim about
# the fitting rule, not about optimiser luck.
CENTRES = np.array([
    [0.0, 0.0, 0.0, 0.0, 0.0],
    [3.0, 1.0, 0.0, -1.0, 2.0],
    [-2.0, 3.0, 1.0, 0.0, -1.0],
])

# A fixed, decidedly non-trivial offset: no zero entries, mixed signs, and a
# norm comparable to the between-centre distances, so the target-side effect is
# a genuine relocation rather than a perturbation.
DELTA = np.array([1.3, -0.7, 2.1, 0.4, -1.9])


def _blobs(rng, n_per_class):
    points = np.vstack([c + 0.6 * rng.normal(size=(n_per_class, len(c))) for c in CENTRES])
    labels = np.repeat(np.arange(len(CENTRES)), n_per_class)
    return points, labels


@pytest.fixture(scope="module")
def fixture():
    """Deterministic source/validation split plus both fitted classifiers.

    ``original`` is fitted on the source training set as recorded; ``translated``
    on the same set with ``DELTA`` added to every row. Labels, hyperparameters
    and random state are identical, which is exactly the condition Proposition 1
    assumes.
    """
    rng = np.random.default_rng(20260922)
    x_train, y_train = _blobs(rng, 24)
    x_val, y_val = _blobs(rng, 10)
    return dict(
        x_train=x_train, y_train=y_train, x_val=x_val, y_val=y_val,
        original=SVC(decision_function_shape="ovo", **FIT_PARAMS).fit(x_train, y_train),
        translated=SVC(decision_function_shape="ovo", **FIT_PARAMS).fit(x_train + DELTA, y_train),
    )


# -- 1. the kernel matrices the fit actually consumes ----------------------
def test_training_gram_matrix_is_preserved_by_a_source_translation(fixture):
    """The proposition's premise: k(x_i + d, x_j + d) equals k(x_i, x_j)."""
    x = fixture["x_train"]
    np.testing.assert_allclose(
        rbf_kernel(x + DELTA, x + DELTA, gamma=GAMMA),
        rbf_kernel(x, x, gamma=GAMMA),
        atol=NUMERICAL_ATOL, rtol=0,
    )


def test_validation_cross_kernel_is_preserved_when_both_sides_translate(fixture):
    """Validation entries are invariant because *both* arguments move."""
    x, v = fixture["x_train"], fixture["x_val"]
    np.testing.assert_allclose(
        rbf_kernel(v + DELTA, x + DELTA, gamma=GAMMA),
        rbf_kernel(v, x, gamma=GAMMA),
        atol=NUMERICAL_ATOL, rtol=0,
    )


def test_gram_equality_is_numerical_rather_than_bitwise(fixture):
    """The mathematical identity is exact; this implementation of it is not.

    Guards the distinction the manuscript draws between the proposition and its
    finite realisation. If a future sklearn made this bit-exact the test would
    fail and the claim could be strengthened deliberately rather than by
    accident, which is the point of pinning it.
    """
    x = fixture["x_train"]
    translated = rbf_kernel(x + DELTA, x + DELTA, gamma=GAMMA)
    original = rbf_kernel(x, x, gamma=GAMMA)
    assert not np.array_equal(translated, original)
    assert np.abs(translated - original).max() < NUMERICAL_ATOL


# -- 2. the fitted solution ------------------------------------------------
def test_translated_source_fit_selects_the_same_solution(fixture):
    """Same support set, same dual coefficients, same intercepts.

    The support set is asserted exactly because it is a discrete choice: if the
    fitting rule really depends on the source features only through the Gram
    matrix, it cannot select a different one. The coefficients are asserted at
    the numerical tolerance, not bitwise.
    """
    a, b = fixture["original"], fixture["translated"]
    assert np.array_equal(a.support_, b.support_)
    assert np.array_equal(a.n_support_, b.n_support_)
    np.testing.assert_allclose(b.dual_coef_, a.dual_coef_, atol=NUMERICAL_ATOL, rtol=0)
    np.testing.assert_allclose(b.intercept_, a.intercept_, atol=NUMERICAL_ATOL, rtol=0)


# -- 3. what source validation sees ---------------------------------------
def test_source_validation_predictions_are_identical_under_translation(fixture):
    """The criterion's own evidence is unchanged: identical predicted labels.

    This is the statement that makes the criterion uninformative. Labels are
    compared exactly -- they are discrete, so there is no rounding to excuse.
    """
    a, b = fixture["original"], fixture["translated"]
    assert np.array_equal(b.predict(fixture["x_val"] + DELTA), a.predict(fixture["x_val"]))


def test_source_validation_decision_values_agree_within_numerical_tolerance(fixture):
    a, b = fixture["original"], fixture["translated"]
    np.testing.assert_allclose(
        b.decision_function(fixture["x_val"] + DELTA),
        a.decision_function(fixture["x_val"]),
        atol=NUMERICAL_ATOL, rtol=0,
    )


def test_source_validation_macro_f1_is_unchanged(fixture):
    """The selection statistic itself, not only the predictions behind it.

    The frozen grid selects on ``source_val`` macro-F1, so this is the quantity
    whose exact equality the ledger audit reports.
    """
    from ser.metrics import macro_f1

    a, b = fixture["original"], fixture["translated"]
    names = [str(c) for c in range(len(CENTRES))]
    truth = [str(c) for c in fixture["y_val"]]
    original = macro_f1(truth, [str(c) for c in a.predict(fixture["x_val"])], names)
    translated = macro_f1(truth, [str(c) for c in b.predict(fixture["x_val"] + DELTA)], names)
    assert translated == original


# -- 4. what the target sees ----------------------------------------------
@pytest.mark.parametrize("shape", ["ovo", "ovr"])
def test_target_decision_function_is_the_original_evaluated_at_x_minus_delta(shape):
    """Equation (1): f_delta(x) = f_0(x - delta) on *unshifted* target vectors.

    Both ``decision_function_shape`` settings are covered. ``ovo`` exposes the
    pairwise functions the proof is stated over; ``ovr`` is sklearn's default
    presentation of them. The identity has to survive that presentation, since
    it is a property of the fitted expansion rather than of how sklearn arranges
    the output columns, which is what this parametrisation certifies.
    """
    rng = np.random.default_rng(20260922)
    x_train, y_train = _blobs(rng, 24)
    _blobs(rng, 10)  # keep the random stream aligned with the module fixture
    a = SVC(decision_function_shape=shape, **FIT_PARAMS).fit(x_train, y_train)
    b = SVC(decision_function_shape=shape, **FIT_PARAMS).fit(x_train + DELTA, y_train)

    target = rng.normal(size=(200, x_train.shape[1])) * 2.0
    np.testing.assert_allclose(
        b.decision_function(target), a.decision_function(target - DELTA),
        atol=NUMERICAL_ATOL, rtol=0,
    )
    assert np.array_equal(b.predict(target), a.predict(target - DELTA))


def test_every_pairwise_ovo_decision_function_satisfies_the_identity(fixture):
    """The argument applies separately to each binary decision function.

    Asserted column by column so a failure names the pair rather than being
    absorbed into an aggregate over the three one-versus-one problems.
    """
    a, b = fixture["original"], fixture["translated"]
    rng = np.random.default_rng(7)
    target = rng.normal(size=(120, fixture["x_train"].shape[1])) * 2.0
    translated = b.decision_function(target)
    shifted = a.decision_function(target - DELTA)
    assert translated.shape[1] == len(CENTRES) * (len(CENTRES) - 1) // 2
    for column in range(translated.shape[1]):
        np.testing.assert_allclose(
            translated[:, column], shifted[:, column], atol=NUMERICAL_ATOL, rtol=0,
            err_msg=f"one-versus-one decision function {column} is not the translated original",
        )


def test_binary_case_satisfies_the_identity_and_changes_target_predictions():
    """The two-class case, where the ovo/ovr distinction does not arise."""
    rng = np.random.default_rng(31)
    x_train, y_train = _blobs(rng, 24)
    binary = y_train < 2
    x_train, y_train = x_train[binary], y_train[binary]
    a = SVC(**FIT_PARAMS).fit(x_train, y_train)
    b = SVC(**FIT_PARAMS).fit(x_train + DELTA, y_train)

    target = rng.normal(size=(200, x_train.shape[1])) * 2.0
    np.testing.assert_allclose(
        b.decision_function(target), a.decision_function(target - DELTA),
        atol=NUMERICAL_ATOL, rtol=0,
    )
    assert (a.predict(target) != b.predict(target)).any()


# -- 5. the consequence: target predictions can change ---------------------
def _boundary_crossings(a, b):
    """Deterministically scan the segments between class centres.

    Not a random probe. Each segment runs from one class centre to another and
    therefore crosses a decision boundary by construction, so translating it by
    ``DELTA`` is guaranteed to produce points where ``x`` and ``x - delta`` sit
    on opposite sides. Returning every crossing lets the caller assert both that
    at least one exists and what happens at it.
    """
    found = []
    for i, j in ((0, 1), (1, 2), (0, 2)):
        steps = np.linspace(0.0, 1.0, 41)
        points = CENTRES[i] + steps[:, None] * (CENTRES[j] - CENTRES[i]) + DELTA
        changed = a.predict(points) != b.predict(points)
        found.extend(points[changed])
    return np.asarray(found)


def test_a_constructed_target_point_changes_class_under_the_translated_source(fixture):
    """The prediction change is real, and it happens for the stated reason.

    Proposition 1 says the target function is the original evaluated at
    ``x - delta``; a prediction therefore changes exactly when that shifted
    argument lands in a different region. Both halves are asserted here, so the
    test certifies the mechanism and not merely that two classifiers disagree.
    """
    a, b = fixture["original"], fixture["translated"]
    crossings = _boundary_crossings(a, b)
    assert len(crossings) > 0, "constructed segments produced no boundary crossing"

    point = crossings[0][None, :]
    assert b.predict(point)[0] != a.predict(point)[0]
    assert b.predict(point)[0] == a.predict(point - DELTA)[0]


def test_the_criterion_cannot_separate_what_the_target_separates(fixture):
    """The two facts together, which is what makes this a selection failure.

    Identical source-validation evidence, different target predictions, in one
    assertion pair. A reviewer reading only this test sees the whole claim.
    """
    a, b = fixture["original"], fixture["translated"]
    assert np.array_equal(b.predict(fixture["x_val"] + DELTA), a.predict(fixture["x_val"]))
    assert len(_boundary_crossings(a, b)) > 0
