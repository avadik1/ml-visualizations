"""Стандартный RoPE: соседние пары, положительное вращение столбцов."""

from numbers import Integral, Real

import numpy as np


def frequencies(d, base=10000.0):
    """Частоты theta_i в радианах на позицию, shape (d/2,), float64."""
    if isinstance(d, bool) or not isinstance(d, Integral) or d < 2 or d % 2:
        raise ValueError("d должна быть положительной чётной целой размерностью ≥ 2")
    if not isinstance(base, Real) or not np.isfinite(base) or base <= 1:
        raise ValueError("base должна быть конечным числом > 1")
    return np.float64(base) ** (-2.0 * np.arange(d // 2, dtype=np.float64) / d)


def vector(value):
    """Проверить один вещественный вектор, не допуская скрытого broadcast."""
    raw = np.asarray(value)
    if raw.ndim != 1 or raw.dtype.kind not in "iuf":
        raise ValueError("Ожидается одномерный вещественный вектор shape (d,)")
    result = raw.astype(np.float64)
    frequencies(result.size)
    if not np.all(np.isfinite(result)):
        raise ValueError("Координаты вектора должны быть конечными")
    return result


def rotate(value, position, base=10000.0):
    """R_p @ x; shape (d,) → (d,), float64.

    Пары хранятся строками, поэтому эквивалент матричной операции —
    pairs @ R(alpha).T, а не pairs @ R(alpha).
    """
    x = vector(value)
    if not isinstance(position, Real) or not np.isfinite(position):
        raise ValueError("Позиция должна быть конечным вещественным числом")
    angles = position * frequencies(x.size, base)
    pairs = x.reshape(-1, 2)
    cos, sin = np.cos(angles), np.sin(angles)
    return np.column_stack((pairs[:, 0] * cos - pairs[:, 1] * sin,
                            pairs[:, 0] * sin + pairs[:, 1] * cos)).ravel()


def scores(q, k, m, n, c=0, base=10000.0):
    """Полные dot products: прямой, относительный и с общим сдвигом."""
    q, k = vector(q), vector(k)
    if q.shape != k.shape:
        raise ValueError("q и k должны иметь одинаковую форму (d,)")
    # Сначала проверяем отдельные позиции (включая c), затем их суммы.
    qm, kn = rotate(q, m, base), rotate(k, n, base)
    rotate(q, c, base)
    return {
        "direct": float(qm @ kn),
        "relative": float(q @ rotate(k, n - m, base)),
        "shifted": float(rotate(q, m + c, base) @ rotate(k, n + c, base)),
    }


def synthetic_vectors(d, seed):
    frequencies(d)
    if isinstance(seed, bool) or not isinstance(seed, Integral) or not 0 <= seed < 2**32:
        raise ValueError("seed должен быть целым числом от 0 до 2**32 − 1")
    rng = np.random.default_rng(seed)
    return rng.normal(size=d).astype(np.float64), rng.normal(size=d).astype(np.float64)
