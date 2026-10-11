# Case study: 2013-02-01

`profile_line_box_20130201.png/pdf/svg` is the paper's P1–P6 profile figure, using predictions aggregated across three UNC models. The light-blue shading is the calibrated **68% predictive interval (q16–q84)**, labeled `UNC predictive 68% interval` in the legend.

`profile_line_box_20130201_source_data.csv` contains 33 observations, point predictions, and q025/q16/q84/q975 for effective radius and SSA. These quantiles provide the 95% and 68% intervals. Each model contributes 256 samples, for 768 samples per observation. The kappa radius convention and original P1–P6 mapping are preserved. Metadata and the caption use the same filename prefix.

Run the following command from the release root to redraw the figure:

```powershell
.\.venv\Scripts\python.exe examples/redraw_paper_profiles.py
```

The result is written to `outputs/paper_profiles_redrawn`, using frozen data and the original plotting function. No inference is required. Other daily curtains and matching artifacts retain their archived provenance.
