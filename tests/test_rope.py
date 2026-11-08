"""Независимые эталоны для соглашения R(alpha) @ вектор-столбец."""

import math

import numpy as np
import pytest
from numpy.testing import assert_allclose

from ml_visualizations.rope import frequencies, rotate, scores, synthetic_vectors

# float64; d ≤ 32, |позиция| ≤ 1024. Запас к округлению sin/cos и суммированию.
TOL = dict(rtol=1e-11, atol=1e-12)


def matrix_reference(x, position, base):
    result = []
    d = len(x)
    for i in range(d // 2):
        angle = position * base ** (-2 * i / d)
        rotation = np.array([[math.cos(angle), -math.sin(angle)],
                             [math.sin(angle), math.cos(angle)]], dtype=np.float64)
        result.extend(rotation @ np.array(x[2*i:2*i+2], dtype=np.float64))
    return np.array(result)


@pytest.mark.parametrize("d", [2, 8, 16, 32])
@pytest.mark.parametrize("base", [10.0, 10000.0, 1e6])
def test_rotation_invariants_and_independent_matrix(d, base):
    q, _ = synthetic_vectors(d, 42)
    assert_allclose(rotate(q, 0, base), q, rtol=0, atol=0)
    for p in [-1024, -7, 0.5, 1, 127, 1024]:
        rotated = rotate(q, p, base)
        assert rotated.dtype == np.float64
        assert_allclose(rotated, matrix_reference(q, p, base), **TOL)
        assert_allclose(np.linalg.norm(rotated.reshape(-1, 2), axis=1),
                        np.linalg.norm(q.reshape(-1, 2), axis=1), **TOL)
        assert_allclose(np.linalg.norm(rotated), np.linalg.norm(q), **TOL)


@pytest.mark.parametrize("d", [2, 8, 32])
@pytest.mark.parametrize("m,n,c", [(0, 1, 0), (-128, 128, 256),
                                   (128, -128, -256), (7, 7, 100)])
def test_relative_formula_and_common_shift(d, m, n, c):
    q, k = synthetic_vectors(d, 5)
    before = q.copy(), k.copy()
    values = scores(q, k, m, n, c)
    independent = matrix_reference(q, m, 10000.0) @ matrix_reference(k, n, 10000.0)
    assert_allclose(list(values.values()), independent, **TOL)
    assert_allclose(q, before[0], rtol=0, atol=0)
    assert_allclose(k, before[1], rtol=0, atol=0)


def test_sign_reference():
    result = scores([1, 0], [0, 1], 0, 1)
    assert_allclose(list(result.values()), -math.sin(1), **TOL)


def test_frequency_definition_and_pair_sum():
    assert_allclose(frequencies(8), [1, 0.1, 0.01, 0.001], **TOL)
    q, k = synthetic_vectors(8, 42)
    qr, kr = rotate(q, 3).reshape(-1, 2), rotate(k, 11).reshape(-1, 2)
    assert_allclose(np.sum(qr * kr), scores(q, k, 3, 11)["direct"], **TOL)


@pytest.mark.parametrize("d", [0, -2, 1, 3, 8.0, True])
def test_invalid_dimension(d):
    with pytest.raises(ValueError, match="чётной"):
        frequencies(d)


@pytest.mark.parametrize("x", [[], [1], [1, 2, 3], [[1, 2]], 3,
                              [1, float("nan")], [1, float("inf")], [1j, 2], ["1", "2"]])
def test_invalid_vector(x):
    with pytest.raises(ValueError):
        rotate(x, 1)


@pytest.mark.parametrize("base", [0, 1, -1, float("nan"), float("inf"), "bad"])
def test_invalid_base(base):
    with pytest.raises(ValueError, match="base"):
        frequencies(8, base)


@pytest.mark.parametrize("p", [float("nan"), float("inf"), [1], "1"])
def test_invalid_position(p):
    with pytest.raises(ValueError, match="Позиция"):
        rotate([1, 0], p)


def test_mismatched_q_k():
    with pytest.raises(ValueError, match="одинаковую форму"):
        scores([1, 2], [1, 2, 3, 4], 0, 1)


def test_reproducible_vectors():
    a = synthetic_vectors(8, 42)
    b = synthetic_vectors(8, 42)
    assert_allclose(a, b, rtol=0, atol=0)
    assert not np.array_equal(a[0], synthetic_vectors(8, 43)[0])


@pytest.mark.parametrize("seed", [-1, 2**32, 1.5, True])
def test_invalid_seed(seed):
    with pytest.raises(ValueError, match="seed"):
        synthetic_vectors(8, seed)
