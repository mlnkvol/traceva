import numpy as np
from typing import List, Tuple
from scipy.special import comb


def fit_cubic_bezier(points: np.ndarray, tolerance: float = 1.0) -> List[Tuple]:
    """
    Апроксимує полілінію кубічними кривими Безьє.
    Повертає список (P0, P1, P2, P3) — control points.
    """
    if len(points) < 2:
        return []

    segments = []
    _fit_recursive(points, tolerance, segments)
    return segments


def _fit_recursive(pts: np.ndarray, tol: float, out: List):
    n = len(pts)
    if n == 2:
        # Пряма лінія як Безьє
        p0, p3 = pts[0], pts[-1]
        p1 = p0 + (p3 - p0) / 3
        p2 = p0 + 2 * (p3 - p0) / 3
        out.append((p0, p1, p2, p3))
        return

    # Параметризація за довжиною дуги
    t = _chord_length_param(pts)

    # Підбираємо контрольні точки методом найменших квадратів
    p0, p3 = pts[0].astype(float), pts[-1].astype(float)
    A, b_vec = _build_lsq(pts, t, p0, p3)

    try:
        result = np.linalg.lstsq(A, b_vec, rcond=None)
        ctrl = result[0]
        p1, p2 = ctrl[0:2], ctrl[2:4]
    except np.linalg.LinAlgError:
        p1 = p0 + (p3 - p0) / 3
        p2 = p0 + 2 * (p3 - p0) / 3

    # Перевіряємо максимальне відхилення
    max_err, split_idx = _max_error(pts, t, p0, p1, p2, p3)

    if max_err < tol:
        out.append((p0, p1, p2, p3))
    else:
        # Рекурсивно ділимо
        mid = max(1, min(split_idx, n - 2))
        _fit_recursive(pts[:mid + 1], tol, out)
        _fit_recursive(pts[mid:], tol, out)


def _chord_length_param(pts: np.ndarray) -> np.ndarray:
    diffs = np.diff(pts, axis=0)
    dists = np.linalg.norm(diffs, axis=1)
    dists = np.concatenate([[0], np.cumsum(dists)])
    total = dists[-1]
    if total == 0:
        return np.linspace(0, 1, len(pts))
    return dists / total


def _bezier_point(t: float, p0, p1, p2, p3) -> np.ndarray:
    return (
        (1 - t) ** 3 * p0 +
        3 * (1 - t) ** 2 * t * p1 +
        3 * (1 - t) * t ** 2 * p2 +
        t ** 3 * p3
    )


def _build_lsq(pts, t, p0, p3):
    n = len(pts)
    A = np.zeros((n, 2, 2))
    b_vec = np.zeros((n, 2))

    for i, ti in enumerate(t):
        b1 = 3 * (1 - ti) ** 2 * ti
        b2 = 3 * (1 - ti) * ti ** 2
        A[i, :, 0] = b1
        A[i, :, 1] = b2

        b3 = (1 - ti) ** 3
        b4 = ti ** 3
        b_vec[i] = pts[i] - b3 * p0 - b4 * p3

    A_flat = np.zeros((2 * n, 4))
    b_flat = np.zeros(2 * n)
    for i in range(n):
        A_flat[2 * i,   0] = A[i, 0, 0]
        A_flat[2 * i,   2] = A[i, 0, 1]
        A_flat[2 * i+1, 1] = A[i, 1, 0]
        A_flat[2 * i+1, 3] = A[i, 1, 1]
        b_flat[2 * i]   = b_vec[i, 0]
        b_flat[2 * i+1] = b_vec[i, 1]

    return A_flat, b_flat


def _max_error(pts, t, p0, p1, p2, p3):
    max_err = 0.0
    split_idx = len(pts) // 2
    for i, ti in enumerate(t):
        bpt = _bezier_point(ti, p0, p1, p2, p3)
        err = np.linalg.norm(pts[i] - bpt)
        if err > max_err:
            max_err = err
            split_idx = i
    return max_err, split_idx
