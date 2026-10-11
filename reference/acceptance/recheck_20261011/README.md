# Independent archive recheck: 2026-10-11

Both complete archives passed bounded CPU verification in their own installed environments. Each inference check prohibited file access to the original project, original NASA/experiment directories, and the sibling archive. Training, optimizer steps, and GPU initialization were blocked. The resource policy was two compute threads and one worker, with checks run sequentially.

| Check | Scope and result |
| --- | --- |
| Asset integrity | All 26 locked assets in each archive passed full-file SHA-256 verification |
| Development provenance | All 3360 retained files matched their original full-file SHA-256 records |
| Public interface | Actual CSV command preserved identifiers and order; FAST, ORI, UNC and all ten uncertainty outputs matched frozen results |
| Independent inference | Four NASA observations per archive, including individual-seed and ensemble predictions; 16 uncertainty draws per model |
| Default uncertainty | One observation per archive; 256 draws per model, three models; all ten outputs and 68%/95% bounds |
| Cross-archive consistency | Saved predictions, intervals, and CSV values matched exactly; both smoke and P1-P6 PNG redraws were byte-identical |
| Regression | 11 checks passed with training and GPU guards |
| Original research code | 133 comparisons passed on four NASA and four synthetic observations; six models, forward calculations, FAST and uncertainty |
| Paper tables | Tables II, III and IV formatted cells matched exactly; numeric summaries passed the declared 1e-12 tolerance |
| Development preflight | Working Python files parsed and the formal trainer imported; no training invoked |

The maximum difference from the archived reference CSV values was 4.547473508864641e-13. Cross-archive predictions and intervals had a maximum difference of zero. P1-P6 figures were redrawn from frozen 33-observation data with the original uncertainty plot function. The smoke plots visibly include the predictive uncertainty shading for effective radius, SSA and N.

The development archive contains the user-facing inference code, complete assets, examples, validation code and reference results, plus development workflows and original experiment records. Public root README/package/Git-ignore templates are stored under `reference/`; the development root uses its own development documentation and package metadata. License and Git text settings are included in the fixed export specification. Environments, generated outputs and Git history remain local to each archive.

This recheck did not retrain, recalibrate, rebuild a LUT, execute the complete synthetic test, or reevaluate all 113 NASA observations. Existing per-seed results were reaggregated for the paper tables. The full evaluation commands remain available as documented.

Evidence: [summary](report.json), [user installation](user/report.json), [development installation](development/report.json), [retained file integrity](archive_integrity.json), [original-code comparison](numerical_comparison.json), [regression](regression.xml), [development preflight](development_preflight.json), and [paper tables](paper_reproduction/report.json).
