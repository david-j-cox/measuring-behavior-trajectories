#!/usr/bin/env python3
"""Run S-Map nonlinearity analysis standalone."""
import multiprocessing
multiprocessing.set_start_method("fork", force=True)

import pandas as pd
from analysis_pipeline.utils import load_config
from analysis_pipeline.dynamical_analysis import _run_smap
import os, time

if __name__ == '__main__':
    config = load_config("config.yaml")
    events_df = pd.read_csv("outputs/tables/events_processed.csv")
    output_dir = config.get("output_dir", "./outputs")
    fmt = config.get("plot_format", "png")
    fig_dir = os.path.join(output_dir, "figures", "dynamical")

    tables_dir = os.path.join(output_dir, "tables")
    os.makedirs(tables_dir, exist_ok=True)

    # Pre-compute best_E cache for all sessions
    from analysis_pipeline.dynamical_analysis import _compute_best_E_for_session
    from joblib import Parallel, delayed
    import gc
    session_ids = events_df["session_id"].unique()
    best_E_results = []
    for i in range(0, len(session_ids), 2):
        batch = session_ids[i:i + 2]
        best_E_results.extend(Parallel(n_jobs=1)(
            delayed(_compute_best_E_for_session)(events_df, sid)
            for sid in batch
        ))
        gc.collect()
    best_E_cache = {r[0]: r[1] for r in best_E_results if r is not None}

    print("=== Running S-Map nonlinearity (section 9) ===")
    t0 = time.time()
    smap_results = _run_smap(events_df, config, fig_dir, fmt, tables_dir, best_E_cache)
    elapsed = time.time() - t0
    print(f"  S-Map completed in {elapsed:.1f}s")

    if len(smap_results) > 0:
        smap_results.to_csv(os.path.join(output_dir, "tables", "smap_results.csv"), index=False)
        print(f"  Saved smap_results.csv ({len(smap_results)} sessions)")
        print(f"\n  S-Map summary:")
        print(f"    Mean simplex rho:  {smap_results['simplex_rho'].mean():.3f}")
        print(f"    Mean S-Map rho:    {smap_results['smap_rho'].mean():.3f}")
        print(f"    Mean nonlinearity: {smap_results['nonlinearity'].mean():.3f}")
        print(f"    Mean best theta:   {smap_results['best_theta'].mean():.2f}")
        n_nonlinear = (smap_results['nonlinearity'] > 0).sum()
        print(f"    Sessions with nonlinearity > 0: {n_nonlinear}/{len(smap_results)}")
    else:
        print("  No S-Map results produced.")
