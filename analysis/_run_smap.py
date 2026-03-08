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

    print("=== Running S-Map nonlinearity (section 9) ===")
    t0 = time.time()
    smap_results = _run_smap(events_df, config, fig_dir, fmt)
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
