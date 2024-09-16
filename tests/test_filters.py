import numpy as np
import pytest
from matplotlib import pyplot as plt

from sigpyproc.core.filters import MatchedFilter, Template
from sigpyproc.core.stats import ZScoreResult


@pytest.fixture(scope="module", autouse=True)
def random_data() -> np.ndarray:
    rng = np.random.default_rng(42)
    return rng.normal(loc=0, scale=1, size=1000).astype(np.float32)


@pytest.fixture(scope="module", autouse=True)
def pulse_data() -> np.ndarray:
    rng = np.random.default_rng(42)
    x = np.linspace(-10, 10, 1000, dtype=np.float32)
    pulse = np.exp(-0.5 * (x**2))
    noise = rng.normal(0, 0.1, 1000).astype(np.float32)
    return pulse + noise


class TestMatchedFilter:
    def test_initialization(self, pulse_data: np.ndarray) -> None:
        mf = MatchedFilter(pulse_data, temp_kind="boxcar")
        assert isinstance(mf, MatchedFilter)
        assert mf.data.shape == pulse_data.shape
        assert mf.temp_kind == "boxcar"
        assert isinstance(mf.zscores, ZScoreResult)
        with pytest.raises(ValueError):
            MatchedFilter(np.zeros((2, 2), dtype=np.float32))

    def test_fails(self, pulse_data: np.ndarray) -> None:
        with pytest.raises(ValueError):
            MatchedFilter(pulse_data, temp_kind="gaussian", spacing_factor=1)
        with pytest.raises(ValueError):
            MatchedFilter(np.ones(10), nbins_max=20)

    def test_convolution(self, pulse_data: np.ndarray) -> None:
        mf = MatchedFilter(pulse_data)
        np.testing.assert_equal(mf.convs.shape, (len(mf.temp_widths), len(mf.data)))
        np.testing.assert_equal(mf.best_model.shape, mf.data.shape)
        assert isinstance(mf.best_temp, Template)
        assert mf.snr > 0
        assert 0 <= mf.peak_bin < len(mf.data)
        start, end = mf.on_pulse
        assert 0 <= start < end <= len(mf.data)

    def test_plot(self, pulse_data: np.ndarray) -> None:
        mf = MatchedFilter(pulse_data)
        fig = mf.plot()
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    @pytest.mark.parametrize(("size_max", "spacing_factor"), [(16, 1.2), (64, 2.0)])
    def test_get_box_width_spacing(self, size_max: int, spacing_factor: float) -> None:
        widths = MatchedFilter.get_box_width_spacing(size_max, spacing_factor)
        assert isinstance(widths, np.ndarray)
        assert widths[0] == 1
        assert np.all(np.diff(widths) > 0)
        assert widths[-1] <= size_max


class TestTemplate:
    @pytest.mark.parametrize("width", [1, 5, 10])
    def test_boxcar(self, width: int) -> None:
        temp = Template.gen_boxcar(width)
        assert isinstance(temp, Template)
        assert temp.kind == "boxcar"
        assert temp.width == width
        assert temp.data.size == width
        assert str(temp) == repr(temp)
        assert "Template" in str(temp)

    @pytest.mark.parametrize(("width", "extent"), [(1.0, 3.5), (5.0, 4.0)])
    def test_gaussian(self, width: float, extent: float) -> None:
        temp = Template.gen_gaussian(width, extent)
        assert isinstance(temp, Template)
        assert temp.kind == "gaussian"
        np.testing.assert_equal(temp.width, width)
        expected_size = int(np.ceil(extent * width / 2.355) * 2 + 1)
        np.testing.assert_equal(temp.data.size, expected_size)

    @pytest.mark.parametrize(("width", "extent"), [(1.0, 3.5), (5.0, 4.0)])
    def test_lorentzian(self, width: float, extent: float) -> None:
        temp = Template.gen_lorentzian(width, extent)
        assert isinstance(temp, Template)
        assert temp.kind == "lorentzian"
        np.testing.assert_equal(temp.width, width)
        expected_size = int(np.ceil(extent * width / 2.355) * 2 + 1)
        np.testing.assert_equal(temp.data.size, expected_size)

    def test_fails(self) -> None:
        with pytest.raises(ValueError):
            Template.gen_boxcar(-1)
        with pytest.raises(ValueError):
            Template.gen_gaussian(-1)
        with pytest.raises(ValueError):
            Template.gen_lorentzian(-1)
        with pytest.raises(ValueError):
            Template(np.array([]), 5, 2)
        with pytest.raises(ValueError):
            Template(np.zeros((10, 10)), 5, 2)
        with pytest.raises(ValueError):
            Template(np.zeros(10), 5, 20)

    def test_get_model(self) -> None:
        temp = Template.gen_boxcar(5)
        model = temp.get_model(peak_bin=10, nbins=20)
        np.testing.assert_equal(model.size, 20)
        assert np.all(model[10:15] > 0)

    def test_get_on_pulse(self) -> None:
        temp = Template.gen_boxcar(5)
        on_pulse = temp.get_on_pulse(peak_bin=10, nbins=20)
        np.testing.assert_equal(on_pulse, (10, 15))
        temp = Template.gen_gaussian(5)
        on_pulse = temp.get_on_pulse(peak_bin=10, nbins=20)
        np.testing.assert_equal(on_pulse, (5, 15))

    def test_plot(self) -> None:
        temp = Template.gen_gaussian(5)
        fig = temp.plot()
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


@pytest.mark.parametrize("temp_kind", ["boxcar", "gaussian", "lorentzian"])
def test_end_to_end(pulse_data: np.ndarray, temp_kind: str) -> None:
    mf = MatchedFilter(pulse_data, temp_kind=temp_kind)  # type: ignore[arg-type]
    assert mf.snr > 5  # Assuming a strong pulse
    assert 450 < mf.peak_bin < 550  # Assuming pulse is roughly in the middle
    assert isinstance(mf.best_model, np.ndarray)
