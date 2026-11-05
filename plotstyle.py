"""Shared matplotlib style for publication figures (validated categorical palette, recessive chrome)."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
SEQ_CMAP = LinearSegmentedColormap.from_list("seq_blue", SEQ)
DIV_CMAP = LinearSegmentedColormap.from_list("div_br", ["#184f95", "#3987e5", "#9ec5f4", "#f0efec",
                                                         "#f3a5a4", "#e34948", "#a32a29"])
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]


def setup():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7, "legend.frameon": False,
        "axes.edgecolor": AXIS, "axes.linewidth": 0.6, "axes.labelcolor": INK2, "text.color": INK,
        "xtick.color": MUTED, "ytick.color": MUTED, "xtick.major.width": 0.5, "ytick.major.width": 0.5,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5, "axes.axisbelow": True,
        "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 1.4,
        "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight", "pdf.fonttype": 42,
        "axes.prop_cycle": matplotlib.cycler(color=CAT),
    })


def save(fig, path_noext):
    fig.savefig(path_noext + ".pdf")
    fig.savefig(path_noext + ".png")
    plt.close(fig)
