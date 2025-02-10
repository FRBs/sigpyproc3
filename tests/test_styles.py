from matplotlib import pyplot as plt

from sigpyproc.viz import styles


class TestPlots:
    def test_plot_tables(self) -> None:
        table = styles.PlotTable()
        table.add_entry("test", 1, "s")
        table.skip_line()
        fig, ax = plt.subplots()
        table.plot(ax)
        assert not ax.axison
        assert len(ax.texts) == 3
        plt.close(fig)
