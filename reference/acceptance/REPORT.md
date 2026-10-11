# Archive acceptance report

Release identifier: `paper_numeric_v1`. Acceptance date: 2026-10-10. The user release `PI_AecroFormer` and development archive `Training_part` have independent complete asset copies and CPU environments. The original project was retained in place.

On 2026-10-11, the public case figure `examples/case_20130201/profile_line_box_20130201.png/pdf/svg` was synchronized with the frozen formal uncertainty figure, including the UNC 68% predictive shading and legend. The 33-observation source CSV provides both 68% and 95% bounds. Figure-index paths were corrected to the actual figure subdirectory. This update copied existing figures and data without inference, training, or calibration. Original historical figures remain in the development archive's `reference/case_20130201`.

On 2026-10-11, release documentation and active development guides were translated into English. The export template and document hashes were synchronized.

On 2026-10-11, both archives passed a new independent CPU recheck with access to the original project and sibling archive prohibited. The recheck verified 26 assets per archive and all 3360 retained development files, reran 11 regression checks and 133 original-code comparisons, reaggregated the paper tables, and redrew uncertainty figures. Cross-archive predictions and intervals matched exactly. See the [independent archive recheck](recheck_20261011/README.md) for the executed scope and evidence. No training, recalibration, LUT rebuild, or full evaluation was performed.

## Models and assets

ORI and UNC each use three formal seeds, `20260721/20260722/20260723`, with their specified `best_dynamic_val.pth` checkpoints and original scalers. UNC uses the original per-model calibration temperatures. Each UNC first-layer weight has shape `(512,11)` and its input scaler has 11 features. The public input consists of five raw optical measurements; feature construction is automatic. Four direct predictions and four log variances are propagated through the original forward calculation to all ten outputs.

- Original development archive: 3360 files, 28.41 GiB, with full-file SHA-256, provenance, and purpose records.
- Computational release assets: 26 items; development and user copies are verified with full-file SHA-256 during export.
- All 21 checkpoint, scaler, and temperature files were compared with the original sources after verification and remained byte-identical.
- The NASA input contains 113 observations exported by the original QC functions. Raw observations, matching intermediates, processed data, and the 2013-02-01 case are archived.
- Synthetic baseline: original 500000-row subset, seed 20260415, K=64, G11 bank-standard-deviation normalization, inverse-distance weights.
- NASA FAST: 6291558-row bank, candidate fraction 0.01, 500 trees, omega=0.30, seed 20260412, unweighted tree mean, RNG reset per profile.

See [release assets](../../manifest.json), [full-file hashes](../release_files.json), [asset catalog](../asset_catalog.json), and [figure index](../figure_index.json). Original development provenance is recorded in `reference/archive_manifest.json` in the development archive.

## Executed checks and results

| Check | Actual scope | Result |
| --- | --- | --- |
| CPU environment and development entry point | Python 3.10.18, torch 2.8.0+cpu, scikit-learn 1.6.1; formal trainer import and train --preflight-only | Passed; no training invoked |
| Regression suite | Single/batch inference, six models, aggregation, all intervals including N/V, empty/invalid inputs, FAST profile resets | 11 checks passed |
| Original research code comparison | 4 NASA + 4 synthetic observations; six models; forward optics, SSA, V, N, standard deviations, sampling; two FAST observations | 133 comparisons passed |
| Default UNC uncertainty | One observation; 256 samples per model; seed 20261020; 768-sample mixture; standard deviations and 68%/95% bounds for ten outputs | Matched original sampling and derived formulas |
| Bounded smoke | 4 NASA observations across 3 profiles; FAST, ORI, UNC; 16 samples per UNC model | Finite predictions; individual-seed and ensemble outputs saved; PNG/PDF/CSV generated |
| Independent user installation | Separate .venv; file access to original project, original NASA/experiment directories, and development archive prohibited; actual CSV command and all three modes | Passed; identifiers, order, predictions, and intervals preserved |
| Paper table reaggregation | 150 clean/noisy metric rows and 25 NASA metric rows; original means and sample standard deviations (ddof=1) | Tables II/III/IV formatted cells matched exactly |
| Paper figure redrawing | Frozen P1–P6 data and original plot function; Arial, paper palette, 68% shading | PNG/PDF/SVG redraw passed |

The formal UNC row of Table II remains `mr MAE=0.0494±0.0001`, `rmed MARE=12.93±0.05%`, and `N MARE=51.58±0.91%`. Table IV remains `r_eff MARE=14.59±1.67%` and `SSA MAE=0.0247±0.0032`. These values were reaggregated from existing individual-seed results. Complete test-set inference was not repeated during acceptance.

## Declared tolerances and maximum differences

The tolerances were declared before comparisons: single/batch parameters `atol=1e-7, rtol=1e-6`; original/archive results and sampling propagation for identical batches `atol=1e-12, rtol=1e-12`; FAST requires exact equality. Compiled CPU Mie versus the original pure-Python kernel uses `atol=1e-10, rtol=1e-9`. Table summaries use `atol=rtol=1e-12`; formatted cells must match exactly.

The maximum absolute difference for original/archive identical-batch predictions, forward calculations, FAST outputs, and intervals was 0. The largest single/batch parameter difference was `5.539144642963123e-8`. The compiled CPU and pure-Python Mie kernels matched for the two parameter sets checked. Across all user modes and frozen CSV comparisons, the maximum absolute difference was `4.547473508864641e-13`, within the declared tolerance.

## Execution limits

All computation used CPU, with two compute threads and one worker by default. Models ran sequentially and shared the forward table. Acceptance did not retrain, recalibrate, rebuild a complete LUT, rerun the complete synthetic test, or run inference on the entire 113-observation NASA cohort. Full workflows, data, original configurations, and historical results are retained. Their full execution time and results were not measured as part of this smoke acceptance.

Evidence: [numerical comparison](numerical_comparison.json), [regression report](regression.xml), [CPU environment](cpu_environment.json), [unchanged original models](models_after_checks.json), [independent installation](installation_check/report.json), [smoke](smoke/report.json), and [paper table reproduction](paper_reproduction/report.json).
