# Frozen references and reproduction

The release identifier is `paper_numeric_v1`. ORI and UNC use `best_dynamic_val.pth` for training seeds `20260721/20260722/20260723`, with the original scalers and per-model calibration files. Checkpoint loading requires a strict architecture match.

The main NASA cohort is `examples/nasa/paper_input_113.csv`, exported from the matched observations by the original quality-control functions. Paper tables use `re_ambient`. The 2013-02-01 case preserves `re_ambient_kappa`, the P1–P6 mapping, and the original UTC/PST time conventions. See [NASA baseline notes](nasa_baseline.md) for matching and quality-control details.

| Evaluation | Frozen baseline |
| --- | --- |
| Synthetic test | Original 500000-row subset; seed 20260415; K=64; G11 bank-standard-deviation normalization; inverse-distance weighting |
| NASA validation | Full 6291558-row bank; candidate fraction 0.01; 500 trees; omega=0.30; seed 20260412; unweighted tree mean; RNG reset per profile |

`reference/paper_tables/` contains individual-seed metrics and summary tables. Synthetic metrics are aggregated over three training seeds and three independent test seeds, using sample standard deviation (`ddof=1`). NASA uses three training seeds evaluated on the same observations. Ensemble predictions are saved separately; their error metrics must not be substituted for the mean of individual-seed error metrics.

`preflight --full-hash` verifies every release asset with full-file SHA-256. `smoke` checks bounded inference and plotting; `validate --full` evaluates the complete NASA cohort. The separate development archive provides full synthetic evaluation, training, lookup-table generation, and raw NASA matching workflows.

Figures use Arial and the paper palette: FAST `#C67C2A`, ORI `#6F7F8F`, UNC `#5FA8D3`, and observations `#202020`. New plots save PNG, PDF, and source CSV files. The P1–P6 case figure includes a shaded 68% predictive interval; its source CSV also contains the 95% bounds.
