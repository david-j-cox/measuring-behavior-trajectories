"""Individual differences analysis: PCA + GMM clustering of behavioral phenotypes."""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.stats import f_oneway, kruskal


def run_individual_differences(metrics_df: pd.DataFrame,
                               analysis_results: dict,
                               config: dict,
                               output_dir: str) -> dict:
    """Run PCA + GMM clustering on session-level behavioral features."""
    results = {}
    fmt = config.get("plot_format", "png")
    fig_dir = os.path.join(output_dir, "figures", "individual_differences")
    os.makedirs(fig_dir, exist_ok=True)

    # Step 1: Assemble feature matrix
    feature_df = _assemble_features(metrics_df, analysis_results)
    if feature_df is None or len(feature_df) < 6:
        print("  Not enough sessions for clustering (need >= 6).")
        return results

    results["feature_matrix"] = feature_df
    session_ids = feature_df["session_id"].values
    feature_cols = [c for c in feature_df.columns if c != "session_id"]
    X_raw = feature_df[feature_cols].values

    print(f"  Assembled {len(feature_cols)} features for {len(feature_df)} sessions.")

    # Step 2: Standardize
    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw)

    # Step 3: PCA
    n_components = min(len(feature_cols), len(feature_df) - 1)
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)

    # Keep components explaining >= 90% variance
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    n_keep = max(2, np.searchsorted(cumvar, 0.90) + 1)
    n_keep = min(n_keep, n_components)
    X_reduced = X_pca[:, :n_keep]

    print(f"  PCA: {n_keep} components explain {cumvar[n_keep-1]*100:.1f}% variance.")

    pca_df = pd.DataFrame(X_reduced,
                          columns=[f"PC{i+1}" for i in range(n_keep)])
    pca_df["session_id"] = session_ids
    results["pca"] = pca_df
    results["pca_model"] = pca
    results["pca_n_keep"] = n_keep
    results["explained_variance"] = pca.explained_variance_ratio_
    results["feature_names"] = feature_cols

    # Step 3b: Test multivariate normality of PCA-reduced features
    normality = _test_multivariate_normality(X_reduced)
    results["normality_test"] = normality
    if normality:
        print(f"  Normality (Henze-Zirkler): stat={normality['hz_statistic']:.3f}, "
              f"p={normality['hz_p_value']:.4f}. {normality['interpretation']}")

    # Step 4: GMM with BIC model selection
    max_k = min(6, len(feature_df) // 3)
    max_k = max(max_k, 2)
    k_range = range(2, max_k + 1)
    bic_scores = []
    gmm_models = []

    for k in k_range:
        gmm = GaussianMixture(n_components=k, covariance_type="full",
                               n_init=10, random_state=42)
        gmm.fit(X_reduced)
        bic_scores.append(gmm.bic(X_reduced))
        gmm_models.append(gmm)

    best_idx = np.argmin(bic_scores)
    best_k = list(k_range)[best_idx]
    best_gmm = gmm_models[best_idx]

    labels = best_gmm.predict(X_reduced)
    probs = best_gmm.predict_proba(X_reduced)

    print(f"  GMM: Best k={best_k} (BIC={bic_scores[best_idx]:.1f}).")
    cluster_counts = pd.Series(labels).value_counts().sort_index()
    for cl, count in cluster_counts.items():
        print(f"    Cluster {cl}: {count} sessions")

    results["bic_scores"] = dict(zip(k_range, bic_scores))
    results["best_k"] = best_k

    # Build cluster assignment table
    cluster_df = pd.DataFrame({
        "session_id": session_ids,
        "cluster": labels,
    })
    for k_col in range(best_k):
        cluster_df[f"prob_cluster_{k_col}"] = probs[:, k_col]
    results["cluster_assignments"] = cluster_df

    # Step 5: Hierarchical clustering for dendrogram
    linkage_matrix = linkage(X_reduced, method="ward")
    results["linkage_matrix"] = linkage_matrix

    # Step 6: Cluster profiles (mean standardized features per cluster)
    profile_rows = []
    for cl in range(best_k):
        mask = labels == cl
        means = X[mask].mean(axis=0)
        for j, feat in enumerate(feature_cols):
            profile_rows.append({
                "cluster": cl,
                "feature": feat,
                "mean_z": means[j],
            })
    profile_df = pd.DataFrame(profile_rows)
    results["cluster_profiles"] = profile_df

    # Step 7: Test cluster → total_rewards
    reward_col = None
    for col_name in ["total_rewards", "final_score", "overall_reward_rate"]:
        if col_name in metrics_df.columns:
            reward_col = col_name
            break

    reward_test = {}
    if reward_col:
        merged = cluster_df.merge(metrics_df[["session_id", reward_col]], on="session_id")
        groups = [merged.loc[merged["cluster"] == cl, reward_col].values
                  for cl in range(best_k)]
        groups = [g for g in groups if len(g) >= 2]

        if len(groups) >= 2:
            if all(len(g) >= 5 for g in groups):
                stat, pval = f_oneway(*groups)
                reward_test["test"] = "one-way ANOVA"
            else:
                stat, pval = kruskal(*groups)
                reward_test["test"] = "Kruskal-Wallis"
            reward_test["statistic"] = stat
            reward_test["p_value"] = pval
            reward_test["reward_metric"] = reward_col
            reward_test["group_means"] = {
                f"cluster_{cl}": float(merged.loc[merged["cluster"] == cl, reward_col].mean())
                for cl in range(best_k)
            }
            print(f"  Cluster → {reward_col}: {reward_test['test']} "
                  f"stat={stat:.2f}, p={pval:.4f}")
    results["reward_test"] = reward_test

    # Step 7b: Per-cluster normality & covariance diagnostics
    cluster_diagnostics = []
    for cl in range(best_k):
        mask = labels == cl
        X_cl = X_reduced[mask]
        n_cl = mask.sum()
        diag = {"cluster": cl, "n": int(n_cl)}

        if n_cl >= n_keep + 2:
            # Covariance eigenvalues — check for near-spherical structure
            cov = np.cov(X_cl.T)
            eigvals = np.linalg.eigvalsh(cov)
            condition_number = eigvals[-1] / max(eigvals[0], 1e-10)
            diag["cov_condition_number"] = float(condition_number)
            diag["spherical"] = "yes" if condition_number < 10 else "no"

            # Per-cluster Henze-Zirkler
            cl_norm = _test_multivariate_normality(X_cl)
            if cl_norm:
                diag["hz_statistic"] = cl_norm["hz_statistic"]
                diag["hz_p_value"] = cl_norm["hz_p_value"]
                diag["mv_normal"] = "yes" if cl_norm["hz_p_value"] > 0.05 else "no"
        else:
            diag["cov_condition_number"] = np.nan
            diag["spherical"] = "insufficient_n"
            diag["mv_normal"] = "insufficient_n"

        cluster_diagnostics.append(diag)

    diagnostics_df = pd.DataFrame(cluster_diagnostics)
    results["cluster_diagnostics"] = diagnostics_df
    diagnostics_df.to_csv(
        os.path.join(output_dir, "tables", "cluster_diagnostics.csv"), index=False
    )
    print(f"  Cluster diagnostics:")
    for _, row in diagnostics_df.iterrows():
        print(f"    Cluster {int(row['cluster'])}: n={int(row['n'])}, "
              f"condition#={row.get('cov_condition_number', 'N/A'):.1f}, "
              f"spherical={row.get('spherical', 'N/A')}, "
              f"MV-normal={row.get('mv_normal', 'N/A')}")

    # Step 8: Generate plots
    _plot_pca_variance(pca, n_keep, fig_dir, fmt, config)
    _plot_bic_curve(k_range, bic_scores, best_k, fig_dir, fmt, config)
    _plot_pca_biplot(X_pca, labels, pca, feature_cols, n_keep, session_ids,
                     fig_dir, fmt, config)
    _plot_cluster_profiles(profile_df, best_k, feature_cols, fig_dir, fmt, config)
    _plot_dendrogram(linkage_matrix, session_ids, labels, fig_dir, fmt, config)
    if reward_col:
        _plot_cluster_rewards(cluster_df, metrics_df, reward_col, best_k,
                              fig_dir, fmt, config)

    # Save tables
    table_dir = os.path.join(output_dir, "tables")
    feature_df.to_csv(os.path.join(table_dir, "individual_differences_features.csv"),
                      index=False)
    cluster_df.to_csv(os.path.join(table_dir, "cluster_assignments.csv"),
                      index=False)
    profile_df.to_csv(os.path.join(table_dir, "cluster_profiles.csv"),
                      index=False)

    return results


def _assemble_features(metrics_df: pd.DataFrame,
                       analysis_results: dict) -> pd.DataFrame:
    """Pull session-level features from metrics and analysis results."""
    if len(metrics_df) == 0:
        return None

    # Start with core behavioral metrics
    core_cols = [
        "session_id", "overall_switch_rate", "mean_run_length",
        "choice_entropy", "win_stay_rate", "lose_shift_rate",
        "overall_reward_rate", "total_rewards", "choice_prop_a",
        "total_clicks",
    ]
    available = [c for c in core_cols if c in metrics_df.columns]
    df = metrics_df[available].copy()

    # Adaptation lags
    for lag_col in ["adaptation_lag_phase2_s", "adaptation_lag_phase3_s"]:
        if lag_col in metrics_df.columns:
            df[lag_col] = metrics_df[lag_col].values

    # DFA
    dfa_df = analysis_results.get("dfa")
    if isinstance(dfa_df, pd.DataFrame) and len(dfa_df) > 0:
        full_dfa = dfa_df[dfa_df["scope"] == "full_session"][["session_id", "dfa_alpha"]]
        df = df.merge(full_dfa, on="session_id", how="left")

    # Sample entropy
    se_df = analysis_results.get("sample_entropy")
    if isinstance(se_df, pd.DataFrame) and len(se_df) > 0:
        full_se = se_df[se_df["scope"] == "full_session"][["session_id", "sample_entropy"]]
        df = df.merge(full_se, on="session_id", how="left")

    # RQA
    rqa_df = analysis_results.get("rqa")
    if isinstance(rqa_df, pd.DataFrame) and len(rqa_df) > 0:
        full_rqa = rqa_df[rqa_df["scope"] == "full_session"][
            ["session_id", "determinism", "laminarity", "trapping_time"]
        ].copy()
        for col in ["determinism", "laminarity", "trapping_time"]:
            if col not in full_rqa.columns:
                full_rqa[col] = np.nan
        df = df.merge(full_rqa, on="session_id", how="left")

    # EDM simplex
    edm_df = analysis_results.get("edm_simplex")
    if isinstance(edm_df, pd.DataFrame) and len(edm_df) > 0:
        edm_cols = ["session_id"]
        if "rho" in edm_df.columns:
            edm_cols.append("rho")
        if "best_E" in edm_df.columns:
            edm_cols.append("best_E")
        df = df.merge(edm_df[edm_cols], on="session_id", how="left")

    # S-Map nonlinearity
    smap_df = analysis_results.get("smap")
    if isinstance(smap_df, pd.DataFrame) and len(smap_df) > 0:
        smap_cols = ["session_id"]
        if "nonlinearity" in smap_df.columns:
            smap_cols.append("nonlinearity")
        df = df.merge(smap_df[smap_cols], on="session_id", how="left")

    # CCM asymmetry (reward→choice minus choice→reward at max lib size)
    ccm_df = analysis_results.get("ccm")
    if isinstance(ccm_df, pd.DataFrame) and len(ccm_df) > 0:
        max_lib = ccm_df.loc[ccm_df.groupby("session_id")["lib_size"].idxmax()]
        if "rho_reward_causes_choice" in max_lib.columns and \
           "rho_choice_causes_reward" in max_lib.columns:
            ccm_asym = max_lib[["session_id"]].copy()
            ccm_asym["ccm_asymmetry"] = (
                max_lib["rho_reward_causes_choice"].values -
                max_lib["rho_choice_causes_reward"].values
            )
            df = df.merge(ccm_asym, on="session_id", how="left")

    # Drop session_id for feature matrix, handle NaNs
    feature_cols = [c for c in df.columns if c != "session_id"]

    # Drop columns that are all NaN
    valid_cols = [c for c in feature_cols if df[c].notna().sum() > len(df) * 0.5]
    # Fill remaining NaNs with column median
    for c in valid_cols:
        if df[c].isna().any():
            df[c] = df[c].fillna(df[c].median())

    return df[["session_id"] + valid_cols]


def _test_multivariate_normality(X):
    """
    Henze-Zirkler test for multivariate normality.

    H0: Data follow a multivariate normal distribution.
    Rejection (p < 0.05) suggests non-normality, which would mean
    the GMM's Gaussian component assumption may be violated.

    Henze & Zirkler (1990), Computational Statistics & Data Analysis.
    """
    n, p = X.shape
    if n < p + 2 or n < 5:
        return None

    try:
        # Center the data
        X_centered = X - X.mean(axis=0)
        S = np.cov(X_centered.T)

        # Regularize if near-singular
        S_reg = S + np.eye(p) * 1e-8
        S_inv = np.linalg.inv(S_reg)

        # Mahalanobis distances
        D = np.array([X_centered[i] @ S_inv @ X_centered[i] for i in range(n)])

        # HZ smoothing parameter
        beta = (1.0 / (2 * p)) * (
            (2 * p + 1) / 4.0
        ) ** (1.0 / (p + 4)) * n ** (1.0 / (p + 4))

        # HZ test statistic
        hz_sum1 = 0.0
        for i in range(n):
            for j in range(n):
                diff = X_centered[i] - X_centered[j]
                hz_sum1 += np.exp(-beta**2 / 2 * diff @ S_inv @ diff)
        hz_sum1 /= n**2

        hz_sum2 = 0.0
        for i in range(n):
            hz_sum2 += np.exp(-beta**2 / (1 + beta**2) * D[i] / 2)
        hz_sum2 *= 2 * (1 + beta**2) ** (-p / 2.0) / n

        hz_stat = hz_sum1 - hz_sum2 + (1 + 2 * beta**2) ** (-p / 2.0)

        # Approximate p-value using lognormal approximation
        # Mean and variance under H0
        mu_hz = 1 - (1 + 2 * beta**2) ** (-p / 2.0) * (
            1 + p * beta**2 / (1 + 2 * beta**2)
            + p * (p + 2) * beta**4 / (2 * (1 + 2 * beta**2)**2)
        )

        # Lognormal approximation for p-value
        from scipy.stats import lognorm
        sigma2_hz = 2 * (1 + 4 * beta**2) ** (-p / 2.0) + \
                     2 * (1 + 2 * beta**2) ** (-p) * (
                         1 + 2 * p * beta**4 / (1 + 2 * beta**2)**2
                         + 3 * p * (p + 2) * beta**8 / (4 * (1 + 2 * beta**2)**4)
                     ) - 4 * (1 + beta**2 + beta**4) ** (-p / 2.0) * (
                         1 + 3 * p * beta**4 / (2 * (1 + beta**2 + beta**4))
                         + p * (p + 2) * beta**8 / (2 * (1 + beta**2 + beta**4)**2)
                     )

        # Simple normal approximation of HZ stat for p-value
        from scipy.stats import norm
        z_hz = (hz_stat - mu_hz) / max(np.sqrt(abs(sigma2_hz) / n), 1e-10)
        p_value = 1 - norm.cdf(z_hz)
        p_value = np.clip(p_value, 0, 1)

        if p_value > 0.05:
            interp = ("Multivariate normality not rejected (p > 0.05); "
                       "GMM Gaussian component assumption is supported.")
        else:
            interp = ("Multivariate normality rejected (p < 0.05); "
                       "GMM results should be interpreted with caution. "
                       "The full-covariance GMM is robust to moderate departures.")

        return {
            "hz_statistic": round(float(hz_stat), 4),
            "hz_p_value": round(float(p_value), 4),
            "interpretation": interp,
        }
    except Exception:
        return None


# ---- Plotting functions ----

def _plot_pca_variance(pca, n_keep, fig_dir, fmt, config):
    """Scree plot with cumulative variance explained."""
    fig, ax = plt.subplots(figsize=(8, 5))
    n = len(pca.explained_variance_ratio_)
    x = np.arange(1, n + 1)

    ax.bar(x, pca.explained_variance_ratio_ * 100, color="steelblue",
           edgecolor="white", alpha=0.8, label="Individual")
    ax.plot(x, np.cumsum(pca.explained_variance_ratio_) * 100,
            "o-", color="darkred", linewidth=2, markersize=6, label="Cumulative")
    ax.axhline(90, color="gray", linestyle="--", alpha=0.5, label="90% threshold")
    ax.axvline(n_keep + 0.5, color="orange", linestyle="--", alpha=0.7,
               label=f"Retained (k={n_keep})")
    ax.set_xlabel("Principal Component")
    ax.set_ylabel("Variance Explained (%)")
    ax.set_title("PCA Scree Plot")
    ax.set_xticks(x)
    ax.legend(fontsize=9)
    plt.tight_layout()
    fig.savefig(os.path.join(fig_dir, f"pca_scree.{fmt}"),
                dpi=config.get("dpi", 150))
    plt.close(fig)


def _plot_bic_curve(k_range, bic_scores, best_k, fig_dir, fmt, config):
    """BIC curve for GMM model selection."""
    fig, ax = plt.subplots(figsize=(7, 5))
    ks = list(k_range)
    ax.plot(ks, bic_scores, "o-", color="steelblue", linewidth=2, markersize=8)
    ax.axvline(best_k, color="red", linestyle="--", linewidth=1.5,
               label=f"Best k={best_k}")
    ax.set_xlabel("Number of Clusters (k)")
    ax.set_ylabel("BIC (lower is better)")
    ax.set_title("GMM Model Selection via BIC")
    ax.set_xticks(ks)
    ax.legend()
    plt.tight_layout()
    fig.savefig(os.path.join(fig_dir, f"gmm_bic.{fmt}"),
                dpi=config.get("dpi", 150))
    plt.close(fig)


def _plot_pca_biplot(X_pca, labels, pca, feature_cols, n_keep, session_ids,
                     fig_dir, fmt, config):
    """PCA biplot: scores colored by cluster with feature loadings."""
    import seaborn as sns

    n_clusters = len(np.unique(labels))
    palette = sns.color_palette("Set2", n_clusters)

    fig, axes = plt.subplots(1, 3, figsize=(21, 6))

    # Left: PC1 vs PC2 scatter
    ax = axes[0]
    for cl in range(n_clusters):
        mask = labels == cl
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
                   color=palette[cl], s=60, alpha=0.8,
                   edgecolors="white", linewidths=0.5,
                   label=f"Cluster {cl} (n={mask.sum()})")
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    ax.set_title("PCA Scores by Cluster")
    ax.legend(fontsize=9)

    # Middle: Loading vectors
    ax2 = axes[1]
    loadings = pca.components_[:2].T  # (n_features, 2)
    for i, feat in enumerate(feature_cols):
        ax2.arrow(0, 0, loadings[i, 0], loadings[i, 1],
                  head_width=0.02, head_length=0.01,
                  fc="steelblue", ec="steelblue", alpha=0.7)
        ax2.text(loadings[i, 0] * 1.12, loadings[i, 1] * 1.12,
                 _short_name(feat), fontsize=7, ha="center", va="center")
    ax2.set_xlabel(f"PC1 loading")
    ax2.set_ylabel(f"PC2 loading")
    ax2.set_title("Feature Loadings")
    # Draw unit circle
    theta = np.linspace(0, 2 * np.pi, 100)
    max_load = np.max(np.abs(loadings)) * 1.2
    ax2.set_xlim(-max_load, max_load)
    ax2.set_ylim(-max_load, max_load)
    ax2.axhline(0, color="gray", linewidth=0.5, alpha=0.3)
    ax2.axvline(0, color="gray", linewidth=0.5, alpha=0.3)

    # Right: Combined biplot — scores + loadings on shared axes
    ax3 = axes[2]
    for cl in range(n_clusters):
        mask = labels == cl
        ax3.scatter(X_pca[mask, 0], X_pca[mask, 1],
                    color=palette[cl], s=60, alpha=0.7,
                    edgecolors="white", linewidths=0.5,
                    label=f"Cluster {cl} (n={mask.sum()})")

    # Scale loadings so arrows are visible alongside scores.
    # Use the ratio of score range to max loading magnitude.
    score_range = max(np.abs(X_pca[:, :2]).max(), 1e-6)
    load_max = max(np.abs(loadings).max(), 1e-6)
    scale = score_range / load_max * 0.8

    ax3_load = ax3.twinx()
    ax3_load_x = ax3.twiny()
    # Remove tick labels on secondary axes to avoid clutter
    ax3_load.set_yticks([])
    ax3_load_x.set_xticks([])

    for i, feat in enumerate(feature_cols):
        dx, dy = loadings[i, 0] * scale, loadings[i, 1] * scale
        ax3.annotate("", xy=(dx, dy), xytext=(0, 0),
                     arrowprops=dict(arrowstyle="->", color="firebrick",
                                     lw=1.5, alpha=0.8))
        ax3.text(dx * 1.10, dy * 1.10, _short_name(feat),
                 fontsize=6.5, ha="center", va="center",
                 color="firebrick", fontweight="bold")

    ax3.axhline(0, color="gray", linewidth=0.5, alpha=0.3)
    ax3.axvline(0, color="gray", linewidth=0.5, alpha=0.3)
    ax3.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    ax3.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    ax3.set_title("Combined Biplot: Scores + Loadings")
    ax3.legend(fontsize=8, loc="lower right")

    plt.tight_layout()
    fig.savefig(os.path.join(fig_dir, f"pca_biplot.{fmt}"),
                dpi=config.get("dpi", 150))
    plt.close(fig)

    # If we have 3+ PCs, also plot PC1 vs PC3
    if n_keep >= 3:
        fig, ax = plt.subplots(figsize=(7, 6))
        for cl in range(n_clusters):
            mask = labels == cl
            ax.scatter(X_pca[mask, 0], X_pca[mask, 2],
                       color=palette[cl], s=60, alpha=0.8,
                       edgecolors="white", linewidths=0.5,
                       label=f"Cluster {cl} (n={mask.sum()})")
        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
        ax.set_ylabel(f"PC3 ({pca.explained_variance_ratio_[2]*100:.1f}%)")
        ax.set_title("PC1 vs PC3 by Cluster")
        ax.legend(fontsize=9)
        plt.tight_layout()
        fig.savefig(os.path.join(fig_dir, f"pca_pc1_pc3.{fmt}"),
                    dpi=config.get("dpi", 150))
        plt.close(fig)


def _plot_cluster_profiles(profile_df, n_clusters, feature_cols,
                           fig_dir, fmt, config):
    """Heatmap of standardized feature means per cluster."""
    import seaborn as sns

    pivot = profile_df.pivot(index="feature", columns="cluster", values="mean_z")
    # Reorder features by absolute loading on cluster difference
    feat_order = pivot.reindex(feature_cols)
    feat_order = feat_order.loc[
        feat_order.std(axis=1).sort_values(ascending=False).index
    ]

    fig, ax = plt.subplots(figsize=(4 + n_clusters * 1.5, max(6, len(feature_cols) * 0.4)))
    sns.heatmap(feat_order, annot=True, fmt=".2f", cmap="RdBu_r",
                center=0, linewidths=0.5, ax=ax,
                yticklabels=[_short_name(f) for f in feat_order.index],
                xticklabels=[f"Cluster {c}" for c in range(n_clusters)])
    ax.set_title("Cluster Profiles (Standardized Feature Means)")
    ax.set_ylabel("")
    plt.tight_layout()
    fig.savefig(os.path.join(fig_dir, f"cluster_profiles.{fmt}"),
                dpi=config.get("dpi", 150))
    plt.close(fig)


def _plot_dendrogram(linkage_matrix, session_ids, labels, fig_dir, fmt, config):
    """Hierarchical clustering dendrogram."""
    import seaborn as sns

    n_clusters = len(np.unique(labels))
    palette = sns.color_palette("Set2", n_clusters)
    color_map = {cl: f"C{cl}" for cl in range(n_clusters)}

    fig, ax = plt.subplots(figsize=(max(10, len(session_ids) * 0.5), 6))

    short_labels = [f"{str(sid)[:8]}" for sid in session_ids]
    dendro = dendrogram(linkage_matrix, labels=short_labels, ax=ax,
                        leaf_rotation=90, leaf_font_size=8,
                        color_threshold=0.7 * max(linkage_matrix[:, 2]))
    ax.set_ylabel("Ward Distance")
    ax.set_title("Hierarchical Clustering Dendrogram")
    plt.tight_layout()
    fig.savefig(os.path.join(fig_dir, f"dendrogram.{fmt}"),
                dpi=config.get("dpi", 150))
    plt.close(fig)


def _plot_cluster_rewards(cluster_df, metrics_df, reward_col, n_clusters,
                          fig_dir, fmt, config):
    """Box plot of reward outcome by cluster."""
    import seaborn as sns

    merged = cluster_df.merge(metrics_df[["session_id", reward_col]], on="session_id")

    palette = sns.color_palette("Set2", n_clusters)
    fig, ax = plt.subplots(figsize=(max(5, n_clusters * 2), 5))
    sns.boxplot(data=merged, x="cluster", y=reward_col, hue="cluster",
                palette=palette, width=0.5, ax=ax, legend=False)
    sns.stripplot(data=merged, x="cluster", y=reward_col, color="black",
                  alpha=0.5, size=5, ax=ax)

    ax.set_xlabel("Cluster")
    ax.set_ylabel(_short_name(reward_col).replace("_", " ").title())
    ax.set_title("Total Rewards by Behavioral Cluster")
    ax.set_xticks(range(n_clusters))
    ax.set_xticklabels([f"Cluster {i}" for i in range(n_clusters)])
    plt.tight_layout()
    fig.savefig(os.path.join(fig_dir, f"cluster_rewards.{fmt}"),
                dpi=config.get("dpi", 150))
    plt.close(fig)


def _short_name(feature_name):
    """Shorten long feature names for plot labels."""
    replacements = {
        "overall_switch_rate": "switch_rate",
        "overall_reward_rate": "reward_rate",
        "adaptation_lag_phase2_s": "adapt_lag_ph2",
        "adaptation_lag_phase3_s": "adapt_lag_ph3",
        "rolling_choice_prop_a_clicks": "choice_prop_a",
        "rho_reward_causes_choice": "ccm_rew→cho",
        "rho_choice_causes_reward": "ccm_cho→rew",
        "ccm_asymmetry": "ccm_asym",
        "entropy_diagonal": "diag_entropy",
        "sample_entropy": "samp_entropy",
        "trapping_time": "trap_time",
        "mean_run_length": "run_length",
        "total_rewards": "total_rew",
        "total_clicks": "n_clicks",
        "choice_entropy": "choice_H",
    }
    return replacements.get(feature_name, feature_name)
