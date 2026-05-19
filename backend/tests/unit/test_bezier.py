import numpy as np

from app.services.bezier import fit_cubic_bezier


def _bezier_point(t, p0, p1, p2, p3):
    return (
        (1 - t) ** 3 * p0
        + 3 * (1 - t) ** 2 * t * p1
        + 3 * (1 - t) * t**2 * p2
        + t**3 * p3
    )


def _sample_segments(segments, samples_per_segment=40):
    points = []
    for p0, p1, p2, p3 in segments:
        for t in np.linspace(0, 1, samples_per_segment):
            points.append(_bezier_point(t, p0, p1, p2, p3))
    return np.asarray(points)


def test_empty_input_returns_empty_list():
    assert fit_cubic_bezier(np.empty((0, 2))) == []


def test_two_points_returns_single_segment():
    points = np.array([[0.0, 0.0], [10.0, 5.0]])

    segments = fit_cubic_bezier(points)

    assert len(segments) == 1
    assert np.allclose(segments[0][0], points[0])
    assert np.allclose(segments[0][3], points[1])


def test_straight_line_low_error():
    points = np.column_stack([np.linspace(0, 100, 20), np.zeros(20)])

    segments = fit_cubic_bezier(points, tolerance=1.0)

    assert len(segments) >= 1
    for p0, _p1, _p2, p3 in segments:
        assert abs(p0[1]) < 1.0
        assert abs(p3[1]) < 1.0


def test_circle_approximation():
    angles = np.linspace(0, 2 * np.pi, 64, endpoint=False)
    points = np.column_stack([50 + 50 * np.cos(angles), 50 + 50 * np.sin(angles)])

    segments = fit_cubic_bezier(points, tolerance=0.5)
    sampled = _sample_segments(segments, samples_per_segment=60)

    assert len(segments) >= 4
    for point in points:
        assert np.min(np.linalg.norm(sampled - point, axis=1)) < 1.0


def test_tolerance_affects_segment_count():
    x = np.linspace(0, 100, 50)
    points = np.column_stack([x, 20 * np.sin(x / 8)])

    strict = fit_cubic_bezier(points, tolerance=0.1)
    loose = fit_cubic_bezier(points, tolerance=5.0)

    assert len(loose) <= len(strict)


def test_segment_format():
    points = np.array([[0.0, 0.0], [20.0, 15.0], [40.0, 0.0]])

    segments = fit_cubic_bezier(points)

    for segment in segments:
        assert isinstance(segment, tuple)
        assert len(segment) == 4
        for point in segment:
            assert point.shape == (2,)
