"""Redraw the paper P1-P6 figure from frozen source data; no inference."""
import shutil
import pandas as pd
from pi_aecroformer.api import default_root
from pi_aecroformer.paper_profiles import plot_profiles

root = default_root()
source = root / "reference/paper_profiles/profile_line_box_20130201_source_data.csv"
destination = root / "outputs/paper_profiles_redrawn"
config = {"profile_figure": {"re_xlim": [.080, .200], "ssa_xlim": [.93, 1.01], "altitude_ylim": [300., 900.]}}
plot_profiles(pd.read_csv(source), config, destination)
shutil.copy2(source, destination / source.name)
print(destination)
