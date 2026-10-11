# California FAST baseline and observation processing

The frozen dataset identifier is `california_fast_all_repro_paperlike_v1`. The matching, retrieval, and plotting workflows use the same observation definitions throughout. New processing runs write to a new dataset directory.

## Matching settings

The processing script is `data/nasa/scripts/discoveraq_matchup_fast_ca.py` in the development archive.

- Process all available California dates.
- Identify spirals from contiguous `ProfileNumber` segments, with altitude-derivative segmentation as a fallback.
- Require duration ≥8 minutes and vertical span ≥1.5 km.
- Search the window from 15 minutes before spiral start to 15 minutes after spiral end.
- Require horizontal distance ≤15 km from the spiral center.
- Align HSRL and in-situ data at 75 m vertical and 1.5 minute temporal resolution.
- Use `AE(355–532)` in `[1.5,2.5]` for the main fine-mode analysis. Record and quality-control `dep532` without making it a hard main-analysis exclusion.
- Direct valid observations require all five HSRL channels and core in-situ quantities, represented by `usable_complete_mask`.

From the development root, run a new matching workflow with:

```powershell
.\.venv\Scripts\python.exe -m aecroformer_training match-nasa --dataset-name reproduced_california_paper_v1
```

Matching products are written under `data/nasa/interim/<dataset-name>` and `data/nasa/processed/<dataset-name>`. Key artifacts are `matchup_table_direct.csv` and `profile_groups_direct/*_direct.parquet`.

## Retrieval quality control

The paper evaluation and profile plots use the same filters:

- Finite `alpha355,beta355,alpha532,beta532,beta1064,rh_ambient`.
- Nonnegative optical measurements.
- `alpha355 >= alpha532` and `alpha532 > 0.01`.
- `beta1064 >= 1e-4`.
- `rh_ambient <= 85`.
- `beta355 > beta532 > beta1064`.
- `ssa532_ambient <= 1.1`; missing SSA values are allowed by this filter.

Inputs are `(5,)` or `(n,5)`, ordered as `[alpha355,beta355,alpha532,beta532,beta1064]`. The release API is `pi_aecroformer.retrieve`; models are `FAST`, `ORI`, and `UNC`. Outputs are `mr,mi,rmed_um,lnsigma,D_mode,sigma,r_eff,SSA,V,N`. See [the API guide](usage.md) for units and uncertainty.

## Frozen cohort statistics

The matched dataset covers eight days and 31 primary direct profiles. There are 121 complete matched observations before retrieval filters and 113 afterward. The original profile plotting run produced 30 figures; one profile was skipped because every observation failed the filters.

| Date | Primary direct profiles |
| --- | --- |
| 2013-01-16 | 1 |
| 2013-01-20 | 3 |
| 2013-01-21 | 8 |
| 2013-01-22 | 1 |
| 2013-01-30 | 6 |
| 2013-01-31 | 3 |
| 2013-02-01 | 6 |
| 2013-02-04 | 3 |

SSA comparisons use the NASA ambient field. Effective-radius tables use the legacy radius convention; the daily case uses the kappa convention. The original evaluation also preserves results for both radius conventions. Archived baseline files are in the development archive's `reference/nasa_baseline`; release per-seed table results are in `reference/paper_tables`.
