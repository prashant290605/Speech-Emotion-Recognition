import copy
import json

import numpy as np
import pytest

from ser.analysis.affine_audit import factor_affine, match_translation_rows


def test_factorisation_logits_penalty_and_distances():
    rng = np.random.default_rng(37)
    a = rng.normal(size=(4, 4))
    a[0] = 0  # Source need not be invertible.
    b = rng.normal(size=(4, 4)) + 4 * np.eye(4)
    s, t = rng.normal(size=(2, 4))
    x, y = rng.normal(size=(2, 8, 4))
    w, q = rng.normal(size=(4, 3)), rng.normal(size=3)
    g = factor_affine(a, s, b, t)
    transported = x @ g.relative_matrix + g.relative_offset
    np.testing.assert_allclose(transported @ b + t, x @ a + s)
    v, u = b @ w, t @ w + q
    np.testing.assert_allclose(transported @ v + u, (x @ a + s) @ w + q)
    np.testing.assert_allclose(y @ v + u, (y @ b + t) @ w + q)
    np.testing.assert_allclose(np.sum(w * w), np.trace(v.T @ g.coefficient_metric @ v))
    delta = x - y
    np.testing.assert_allclose(np.sum((delta @ b) ** 2, axis=1),
                               np.einsum("ni,ij,nj->n", delta, g.distance_metric, delta))


def test_zscore_is_diagonal_transport_in_common_coordinates():
    ds, dt = np.diag([2., 3., 1.]), np.diag([4., 1., 2.])
    ms, mt = np.array([1., 2., 0.]), np.array([4., -1., 3.])
    a, b = np.linalg.inv(ds), np.linalg.inv(dt)
    g = factor_affine(a, -ms @ a, b, -mt @ b)
    np.testing.assert_allclose(g.relative_matrix, a @ dt)
    np.testing.assert_allclose(g.relative_offset, mt - ms @ a @ dt)
    np.testing.assert_allclose(g.coefficient_metric, dt @ dt)


def test_translation_preserves_source_scores_but_can_change_target():
    x, delta, w, intercept = np.array([[-2.], [2.]]), np.array([3.]), np.array([[1.]]), np.array([0.])
    shifted_intercept = intercept - delta @ w
    np.testing.assert_allclose((x + delta) @ w + shifted_intercept, x @ w + intercept)
    assert not np.array_equal(x @ w + shifted_intercept > 0, x @ w + intercept > 0)
    d = x[:, None, :] - x[None, :, :]
    z = x + delta
    np.testing.assert_allclose(np.exp(-np.sum(d ** 2, axis=2)),
                               np.exp(-np.sum((z[:, None] - z[None, :]) ** 2, axis=2)))


@pytest.mark.parametrize("b", [np.zeros((2, 2)), np.array([[1., 0.], [0., np.nan]])])
def test_invalid_target_maps_rejected(b):
    with pytest.raises((ValueError, np.linalg.LinAlgError)):
        factor_affine(np.eye(2), np.zeros(2), b, np.zeros(2))


def rows():
    base = dict(freeze_tag="frozen", blending="none", classifier="svm_rbf", status="ok",
                seed=0, search="same", selection_source_val_macro_f1=.7, macro_f1=.2,
                hyperparams_json=json.dumps({"selected": {"C": 1}}), chance_macro_f1=.16)
    return [dict(base, alignment="none", run_id="a"),
            dict(base, alignment="mean_shift", run_id="b", macro_f1=.4)]


def audit(data):
    return match_translation_rows(data, freeze_tag="frozen", classifiers=["svm_rbf"],
                                  match_fields=["seed", "search"])


def test_join_exact_equality_no_mutation():
    data = rows()
    before = copy.deepcopy(data)
    result = audit(data)[0]
    assert result["validation_equal"] and result["hyperparameters_equal"]
    assert result["target_difference"] == pytest.approx(.2)
    assert data == before
    data[1]["selection_source_val_macro_f1"] += 1e-12
    assert not audit(data)[0]["validation_equal"]


@pytest.mark.parametrize("failure", ["missing", "duplicate", "search", "failed", "nonfinite"])
def test_bad_pairs_fail_closed(failure):
    data = rows()
    if failure == "missing":
        data.pop()
    elif failure == "duplicate":
        data.append(copy.deepcopy(data[0]))
    elif failure == "search":
        data[1]["search"] = "different"
    elif failure == "failed":
        data[1]["status"] = "failed"
    else:
        data[1]["macro_f1"] = float("nan")
    with pytest.raises(ValueError):
        audit(data)
