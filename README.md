# PI_AecroFormer

**Physics-Informed AecroFormer for Stable Aerosol Microphysical Retrieval From Multiwavelength Lidar**

Retrieve aerosol microphysical properties, number concentration N, and predictive uncertainty from three backscatter and two extinction measurements (3β + 2α). The default prediction is an ensemble of three pretrained models. Inference runs on CPU with two compute threads.

## Installation

Use Python 3.10. Open PowerShell in this directory:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-cpu.lock.txt
.\.venv\Scripts\python.exe -m pip install --no-deps -e .
.\run.ps1 preflight
```

The complete local release includes model weights and forward lookup tables. [manifest.json](manifest.json) identifies the assets; [reference/release_files.json](reference/release_files.json) records full-file hashes. Large assets are distributed separately from the source repository. A source-only download requires the matching asset bundle.

## Single observation

```python
from pi_aecroformer import retrieve

# Order: [alpha355, beta355, alpha532, beta532, beta1064]
# Extinction alpha: km^-1; backscatter beta: km^-1 sr^-1
x5 = [0.08, 0.003, 0.04, 0.002, 0.0005]
out = retrieve(x5)
print(out['r_eff'], out['SSA'], out['N'])
print(out['uncertainty']['N'])  # std, q16, q84, q025, q975
```

To use an archived NASA observation:

```powershell
.\.venv\Scripts\python.exe examples/single_sample.py
```

## Batch retrieval

The Python interface accepts an `(n,5)` array and returns arrays in the same observation order. CSV input must contain `alpha355,beta355,alpha532,beta532,beta1064`; existing identifiers and other input columns are preserved.

```powershell
.\run.ps1 batch --input examples/nasa/smoke_input.csv --output outputs/predictions.csv
```

The output includes point predictions and uncertainty columns such as `N_std,N_q16,N_q84,N_q025,N_q975`, plus a separate metadata file. Uncertainty propagation takes longer than point prediction. Use `--no-uncertainty` when only point predictions are needed.

## Models and outputs

- `UNC` (default): three-model ensemble with predictive uncertainty for all outputs.
- `ORI`: original neural baseline. `FAST`: the frozen full-bank FAST baseline.
- Select one model with `retrieve(x5, model_seed=20260721)`. Other seeds are `20260722` and `20260723`.
- Pass the five raw optical measurements directly; feature construction is automatic.

| Output | Meaning | Unit |
| --- | --- | --- |
| mr, mi | Real and imaginary parts of the refractive index | Dimensionless |
| rmed_um | Volume median radius | μm |
| lnsigma, sigma | Logarithmic width and geometric standard deviation | Dimensionless |
| D_mode | Mode diameter of the number distribution | μm |
| r_eff | Effective radius | μm |
| SSA | Single-scattering albedo at 532 nm | Dimensionless |
| V | Volume concentration | μm³/cm³ |
| N | Number concentration | cm⁻³ |

## Validation and figures

```powershell
.\run.ps1 smoke --samples 4 --mc-samples 16
.\run.ps1 validate --full --mc-samples 256
.\run.ps1 preflight --full-hash
```

The smoke command performs bounded inference and plotting. It does not train, recalibrate, or rebuild a lookup table. Full NASA validation uses the archived 113-observation cohort. FAST retains its complete bank and 500 pruning trees. Predictions, source data, and PNG/PDF figures are written to `outputs/`.

`examples/nasa_profile.py` demonstrates profile validation and plotting. `examples/redraw_paper_profiles.py` redraws the paper's P1–P6 figure from frozen data without inference. The case figure at `examples/case_20130201/profile_line_box_20130201.png` includes the UNC 68% predictive interval. See [figure_index.json](reference/figure_index.json) for figure, data, and code associations.

[API guide](docs/usage.md) · [Reproduction guide](docs/reproduction.md) · [Acceptance report](reference/acceptance/REPORT.md)

| Directory | Contents |
| --- | --- |
| src/pi_aecroformer | Public API, frozen model and forward calculations, plotting |
| assets | Weights, scalers, calibration temperatures, lookup tables |
| examples | Single and batch retrieval, NASA profiles, daily case |
| validation | Bounded smoke and regression checks |
| reference | Paper, per-seed results, figure data, acceptance evidence, hashes |
| docs | Units, API usage, reproduction instructions |
| outputs | Newly generated results |
