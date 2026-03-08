# TODO

## Analysis Enhancements

- [ ] **7. HMM state interpretation** — Add summary table/visualization of HMM emission parameters (mean choice rate, reward rate, transition matrix) per state, with descriptive labels (e.g., "exploiting A," "exploring," "exploiting B"). Include in report.
- [ ] **8. Combined PCA biplot** — Add a third panel to the PCA biplot figure that overlays PC scores (colored by cluster) and feature loading vectors on the same axes, showing cluster-feature relationships at a glance.

## EDM/S-Map Performance Optimization

## Speedup strategies for Step 8 (Dynamical Systems Analysis)

Step 8 currently takes ~7 hours for 60 sessions. The goal is to reduce this to ~30-45 minutes.

- [ ] **1. Subsample time series** — Downsample rolling signals to ~200 points before passing to pyEDM (S-Map, CCM, Simplex)
- [ ] **2. Parallelize across sessions** — Wrap the per-session S-Map, CCM, and Simplex loops in `joblib.Parallel`
- [ ] **3. Reduce search grids** — Cap `maxE=6`, reduce CCM `sample` to 20, use 5 library sizes
- [ ] **4. Cache `best_E` per session** — Compute `EmbedDimension` once per session and reuse across Simplex, CCM, and S-Map
- [ ] **5. Persist EDM simplex results** — Save `edm_simplex.csv` alongside `smap_results.csv` and `ccm_results.csv`
- [ ] **6. Session-level result caching** — On future runs, load existing CSVs, skip sessions already computed, only run new ones, then merge
