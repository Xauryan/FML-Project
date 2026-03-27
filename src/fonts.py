import matplotlib
from matplotlib import font_manager

from src.paths import BUNDLED_FONT_PATH


def configure_matplotlib_chinese():
    if not BUNDLED_FONT_PATH.exists():
        matplotlib.rcParams["font.family"] = ["DejaVu Sans"]
        matplotlib.rcParams["font.sans-serif"] = ["DejaVu Sans"]
        matplotlib.rcParams["axes.unicode_minus"] = False
        return "DejaVu Sans"

    font_manager.fontManager.addfont(str(BUNDLED_FONT_PATH))
    name = font_manager.FontProperties(fname=str(BUNDLED_FONT_PATH)).get_name()
    matplotlib.rcParams["font.family"] = [name, "DejaVu Sans"]
    matplotlib.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
    matplotlib.rcParams["axes.unicode_minus"] = False
    return name
