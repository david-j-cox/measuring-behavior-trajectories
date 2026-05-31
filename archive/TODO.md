# TODO

## High Priority (before submission)

- [x] **1. Sample size justification** — All 60 sessions passed validation (0 excluded). Document in manuscript with precision argument.
- [x] **2. Pedagogical framing** — Added plain-language explanations, behavior-analytic connections, and interpretation guidance for all dynamical systems techniques in `results_summary.py` and `manuscript_outline.py`.
- [x] **3. Glossary/primer table** — Created 17-entry glossary mapping dynamical systems terms to behavior-analytic equivalents in `table_generation.py`. Outputs `glossary_dynamical_systems.csv` and `.md`.
- [x] **4. Effect size reporting** — Added Cohen's d to phase transition paired t-tests in `phase_analysis.py`. Effect sizes displayed in plots and HTML report with small/medium/large classification.

## Medium Priority (strengthens the paper)

- [x] **5. Simulated null comparison** — Created `null_comparison.py` with random, static matching, and WSLS generators. Runs DFA, entropy, RQA, BOCPD on null data and compares to real. Opt-in via `--null-comparison` flag.
- [x] **6. Matching law prominence** — Added dedicated Matching Law Analysis section in `reports.py` with sensitivity/bias distributions, undermatching classification, and success/failure framing. Updated `results_summary.py` with canonical-benchmark narrative.
- [x] **7. Sensitivity/robustness checks** — Created `robustness_checks.py` testing rolling window size, DFA window range, and BOCPD hazard rate sensitivity. Opt-in via `--robustness` flag.

## Analysis Enhancements

- [x] **8. HMM state interpretation** — Added `summarize_hmm_states()` in `hmm_models.py` with descriptive labels (Exploiting A/B, Exploring). Integrated into report and pipeline.
- [x] **9. Combined PCA biplot** — Added third panel overlaying PC scores (colored by cluster) and feature loading vectors on shared axes in `individual_differences.py`.
- [x] **10. Open science package** — Added de-identified data export, `config.yaml` copy, `reproduce.sh` script, `requirements.txt`, and `data/README.md` to `export_package.py`.

## EDM/S-Map Performance Optimization

- [x] **11. Subsample time series** — Added `_subsample_signal()` helper, called before all pyEDM operations.
- [x] **12. Parallelize across sessions** — Added `joblib.Parallel` for Simplex, CCM, and S-Map per-session work.
- [x] **13. Reduce search grids** — Capped `maxE=6`, CCM to 5 library sizes, `sample=20`.
- [x] **14. Cache `best_E` per session** — Precompute `EmbedDimension` once per session, reuse across all EDM methods.
- [x] **15. Persist EDM simplex results** — Simplex results now saved to `edm_simplex.csv`.
- [x] **16. Session-level result caching** — All EDM functions check for existing CSVs, skip computed sessions, merge new results.
