import numpy as np
import pytest
from matplotlib import pyplot as plt

from sigpyproc.core.filters import MatchedFilter, Template


@pytest.fixture(scope="module", autouse=True)
def random_data() -> np.ndarray:
    rng = np.random.default_rng(42)
    return rng.normal(loc=0, scale=1, size=1000)


@pytest.fixture(scope="module", autouse=True)
def pulse_data() -> np.ndarray:
    rng = np.random.default_rng(42)
    x = np.linspace(-10, 10, 1000)
    pulse = np.exp(-0.5 * (x**2))
    noise = rng.normal(0, 0.1, 1000)
    return pulse + noise


class TestMatchedFilter:
    def test_initialization(self, pulse_data: np.ndarray) -> None:
        mf = MatchedFilter(pulse_data)
        assert isinstance(mf, MatchedFilter)
        assert mf.data.shape == pulse_data.shape
        assert mf.temp_kind == "boxcar"
        assert mf.noise_method == "iqr"
        with pytest.raises(ValueError):
            MatchedFilter(np.zeros((2, 2)))

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

    @pytest.mark.parametrize(("nbins_max", "spacing_factor"), [(16, 1.2), (64, 2.0)])
    def test_width_spacing(self, nbins_max: int, spacing_factor: float) -> None:
        widths = MatchedFilter.get_width_spacing(nbins_max, spacing_factor)
        assert isinstance(widths, np.ndarray)
        assert widths[0] == 1
        assert np.all(np.diff(widths) > 0)
        assert widths[-1] <= nbins_max


class TestTemplate:
    @pytest.mark.parametrize("width", [1, 5, 10])
    def test_boxcar(self, width: int) -> None:
        temp = Template.gen_boxcar(width)
        assert isinstance(temp, Template)
        assert temp.kind == "boxcar"
        assert temp.width == width
        assert temp.size == width
        assert str(temp) == repr(temp)
        assert "Template" in str(temp)

    @pytest.mark.parametrize(("width", "extent"), [(1.0, 3.5), (5.0, 4.0)])
    def test_gaussian(self, width: float, extent: float) -> None:
        temp = Template.gen_gaussian(width, extent)
        assert isinstance(temp, Template)
        assert temp.kind == "gaussian"
        assert temp.width == width
        assert temp.size == int(np.ceil(extent * width / 2.355) * 2 + 1)

    @pytest.mark.parametrize(("width", "extent"), [(1.0, 3.5), (5.0, 4.0)])
    def test_lorentzian(self, width: float, extent: float) -> None:
        temp = Template.gen_lorentzian(width, extent)
        assert isinstance(temp, Template)
        assert temp.kind == "lorentzian"
        assert temp.width == width
        assert temp.size == int(np.ceil(extent * width / 2.355) * 2 + 1)

    def test_get_padded(self) -> None:
        temp = Template.gen_boxcar(5)
        padded = temp.get_padded(10)
        assert len(padded) == 10
        assert np.all(padded[5:] == 0)
        with pytest.raises(ValueError):
            temp.get_padded(3)

    def test_plot(self) -> None:
        temp = Template.gen_gaussian(5)
        fig = temp.plot()
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


@pytest.mark.parametrize("temp_kind", ["boxcar", "gaussian", "lorentzian"])
def test_end_to_end(pulse_data: np.ndarray, temp_kind: str) -> None:
    mf = MatchedFilter(pulse_data, temp_kind=temp_kind)
    assert mf.snr > 5  # Assuming a strong pulse
    assert 450 < mf.peak_bin < 550  # Assuming pulse is roughly in the middle

