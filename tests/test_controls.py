"""Отдельная проверка событий ipywidgets и пересчёта PNG, без браузера."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from ml_visualizations.rope_demo import RopeDemo, make_figure


@pytest.fixture
def config():
    return dict(d=8, base=10000.0, seed=42, m=3, n=11, c=0, pair=1, delta_range=(-64, 64))


def test_events_reset_and_vector_lifecycle(config, tmp_path):
    demo = RopeDemo(config)
    original_q, original_k = demo.q.copy(), demo.k.copy()
    initial_png = bytes(demo.image.value)
    assert initial_png.startswith(b"\x89PNG")
    for name, value in [("m", -128), ("n", 128), ("c", 256),
                        ("base", 100.0), ("pair", 3), ("delta_range", (-256, 256))]:
        before = bytes(demo.image.value)
        demo.controls[name].value = value
        assert bytes(demo.image.value) != before, name
        assert_allclose(demo.q, original_q, rtol=0, atol=0)
        assert_allclose(demo.k, original_k, rtol=0, atol=0)
        assert_allclose(demo.result["shifted"], demo.result["relative"], rtol=1e-11, atol=1e-12)
    demo.controls["d"].value = 2
    assert demo.q.shape == (2,)
    assert demo.controls["pair"].max == 0
    assert demo.controls["pair"].value == 0
    demo.controls["d"].value = 32
    demo.controls["pair"].value = 15
    assert demo.q.shape == (32,)
    old_q = demo.q.copy()
    demo.controls["seed"].value = 43
    assert not np.array_equal(old_q, demo.q)
    demo.reset_button.click()  # Реальный зарегистрированный обработчик on_click.
    assert demo.state == config
    assert_allclose(demo.q, original_q, rtol=0, atol=0)
    assert_allclose(demo.k, original_k, rtol=0, atol=0)
    assert bytes(demo.image.value) == initial_png
    demo.controls["delta_range"].value = (1, 1)
    assert "Ошибка" in demo.summary.value
    assert bytes(demo.image.value) == b""
    demo.reset_button.click()
    assert bytes(demo.image.value) == initial_png


@pytest.mark.parametrize("change", [{"d": 3}, {"d": 34}, {"base": 1}, {"pair": 4},
                                     {"seed": -1}, {"m": 129}, {"c": -257},
                                     {"delta_range": (2, 1)}, {"delta_range": (-300, 3)}])
def test_invalid_config(config, change):
    with pytest.raises(ValueError):
        RopeDemo(dict(config, **change))


def test_plot_values_follow_math(config):
    demo = RopeDemo(config)
    s = demo.state
    fig = make_figure(demo.q, demo.k, **{k: s[k] for k in ("m", "n", "c", "base", "pair", "delta_range")})
    geometry, frequencies, angles, relative = fig.axes
    assert geometry.get_aspect() == 1.0
    assert frequencies.get_yscale() == "log"
    assert_allclose(angles.lines[0].get_ydata(), [3, .3, .03, .003])
    curve = relative.lines[0]
    x = curve.get_xdata()
    assert_allclose(curve.get_ydata()[x == 8], demo.result["direct"], rtol=1e-11, atol=1e-12)
    assert "Только пара i=1" in demo.details.value
    fig.clear()
