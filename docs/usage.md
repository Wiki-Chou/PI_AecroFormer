# Inputs, outputs, and predictive intervals

`retrieve(x5, mode="UNC", model_seed=None, return_uncertainty=True, mc_samples=256, random_seed=20261020)` accepts a single observation with shape `(5,)` or a batch with shape `(n,5)`. Single-observation outputs are scalars; batch outputs are arrays. `uncertainty` contains standard deviations and quantiles for each output. `metadata` records the release, model seeds, units, and resource settings.

The input order is `[alpha355,beta355,alpha532,beta532,beta1064]`. Extinction is in km⁻¹ and backscatter in km⁻¹ sr⁻¹. Convert measurements expressed per Mm to measurements per km by multiplying by `1e-3`. All channels must be finite and nonnegative. Zero values use the frozen numerical floor of `1e-12`. The API does not apply NASA-specific quality filters; those filters belong to paper validation.

UNC internally constructs five group-L2-normalized components and six extinction/backscatter ratios. ORI uses the five normalized components. Direct parameter predictions are ordered as `[mr,mi,rmed_um,lnsigma]`, with bounds `[1.30,1.70]`, `[0,0.5]`, `[0.05,0.50] μm`, and `[0.38,0.50]`, respectively.

`model_seed=None` selects all three release models. V, N, effective radius, and SSA are derived separately for each model, then averaged. V is recovered from the median ratio of observed optical channels to unit-volume forward optical predictions. Parameter conversions and the 4π backscatter normalization follow the frozen reference implementation.

UNC applies the original calibration temperatures to each model's physical parameter standard deviations. It samples a diagonal Gaussian truncated to the physical bounds, then propagates the samples through all derived outputs. Samples from the three models form an equally weighted mixture. `q16/q84` define the 68% predictive interval; `q025/q975` define the 95% interval. N and V intervals include changes in concentration recovery induced by parameter uncertainty. These intervals exclude aircraft collocation and in-situ reference errors. Derived intervals propagate the parameter calibration; they are not calibrated separately.

The same observation order, model seeds, random seed, and sample count reproduce the same intervals. Changing the sample count changes Monte Carlo quantiles. `Retriever.predict_parameters` exposes the raw and calibrated physical parameter standard deviations.

FAST preserves the original random pruning order and batch behavior. NASA batch inputs should retain `group_id_direct`. The CSV command groups profiles automatically; Python callers should pass `group_ids=...`. Without identifiers, the original single-batch behavior is used. Reordering observations or splitting a profile across calls can change FAST's random selections.

Runtime defaults are CPU, two compute threads, and one worker. Set `PI_AECROFORMER_THREADS=1` before starting Python to reduce compute load; supported values are 1–4. The forward table takes time to load initially and requires approximately 12 GB of RAM. Reuse one `Retriever()` instance across calls to share the table.

For a wheel installation with assets stored elsewhere, use `Retriever(asset_root="path/to/complete/release")` or set `PI_AECROFORMER_HOME` to the complete release directory. Runtime paths resolve within that release.

## Verify an independent installation

Run from the root of either complete archive:

```powershell
.\.venv\Scripts\python.exe validation/check_installation.py --full-hash --smoke --default-uncertainty --out-dir outputs/installation_check
```

This checks all asset hashes, the real CSV command, input identifiers and row order, FAST/ORI/UNC predictions against frozen reference outputs, and all ten UNC uncertainty outputs. It also saves a four-observation smoke figure and checks one observation using the default 256 samples per model. Access to the original project and the sibling archive is prohibited during this check; training and GPU initialization are blocked. Models run sequentially and share one forward table. Results are written to the specified output directory.
