"""Baseline behavioral models: random, bias, WSLS, logistic regression."""

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, accuracy_score


def _fit_session_baselines(sid, sdf):
    """Fit all baseline models for a single session."""
    sdf = sdf.sort_values("timestamp_ms").reset_index(drop=True)
    choices = sdf["choice_a"].values

    if len(choices) < 10:
        return []

    rows = []

    random_nll = -np.sum(np.log(0.5) * np.ones(len(choices)))
    rows.append(_model_row(sid, "random", 0, random_nll, len(choices)))

    bias_result = _fit_bias(choices)
    rows.append(_model_row(sid, "bias", 1, bias_result["nll"], len(choices)))

    if "reward_outcome" in sdf.columns:
        wsls_result = _fit_wsls(choices, sdf["reward_outcome"].values)
        rows.append(_model_row(sid, "wsls", 2, wsls_result["nll"], len(choices)))

    logistic_result = _fit_logistic(sdf)
    if logistic_result is not None:
        rows.append(_model_row(
            sid, "logistic", logistic_result["n_params"],
            logistic_result["nll"], len(choices)
        ))

    return rows


def fit_all_baselines(events_df: pd.DataFrame, config: dict = None) -> pd.DataFrame:
    """Fit all baseline models per session (parallelized)."""
    n_jobs = 2 if config is None else config.get("n_jobs", 2)

    sessions = [(sid, sdf) for sid, sdf in events_df.groupby("session_id")]
    results = Parallel(n_jobs=n_jobs)(
        delayed(_fit_session_baselines)(sid, sdf)
        for sid, sdf in sessions
    )

    rows = [row for session_rows in results for row in session_rows]
    return pd.DataFrame(rows)


def _model_row(session_id, model_name, n_params, nll, n_obs):
    """Create a standardized model comparison row."""
    aic = 2 * n_params + 2 * nll
    bic = n_params * np.log(n_obs) + 2 * nll
    return {
        "session_id": session_id,
        "model": model_name,
        "n_params": n_params,
        "nll": nll,
        "aic": aic,
        "bic": bic,
        "n_obs": n_obs,
    }


def _fit_bias(choices: np.ndarray) -> dict:
    """Fit a constant-bias model: P(A) = p."""
    p = np.clip(np.mean(choices), 1e-6, 1 - 1e-6)
    nll = -np.sum(choices * np.log(p) + (1 - choices) * np.log(1 - p))
    return {"p": p, "nll": nll}


def _fit_wsls(choices: np.ndarray, rewards: np.ndarray) -> dict:
    """
    Fit WSLS model with parameters:
    - p_stay_win: P(stay | win)
    - p_shift_lose: P(shift | lose)
    """
    def neg_log_lik(params):
        p_stay_win, p_shift_lose = params
        p_stay_win = np.clip(p_stay_win, 1e-6, 1 - 1e-6)
        p_shift_lose = np.clip(p_shift_lose, 1e-6, 1 - 1e-6)

        ll = 0.0
        for t in range(1, len(choices)):
            if rewards[t - 1] == 1:
                p_stay = p_stay_win
            else:
                p_stay = 1 - p_shift_lose

            # Likelihood based on whether subject repeated or switched
            if choices[t] == choices[t - 1]:
                ll += np.log(max(p_stay, 1e-10))
            else:
                ll += np.log(max(1 - p_stay, 1e-10))

        return -ll

    result = minimize(neg_log_lik, [0.7, 0.3], bounds=[(0.01, 0.99), (0.01, 0.99)],
                      method="L-BFGS-B")
    return {"p_stay_win": result.x[0], "p_shift_lose": result.x[1], "nll": result.fun}


def _fit_logistic(sdf: pd.DataFrame) -> dict:
    """Logistic regression predicting choice from recent features."""
    feature_cols = []
    X_data = {}

    # Previous choice
    if "choice_a" in sdf.columns:
        X_data["prev_choice"] = sdf["choice_a"].shift(1)
        feature_cols.append("prev_choice")

    # Previous reward
    if "reward_outcome" in sdf.columns:
        X_data["prev_reward"] = sdf["reward_outcome"].shift(1)
        feature_cols.append("prev_reward")

    # Previous choice x reward interaction
    if "prev_choice" in X_data and "prev_reward" in X_data:
        X_data["choice_x_reward"] = X_data["prev_choice"] * X_data["prev_reward"]
        feature_cols.append("choice_x_reward")

    # Run length
    if "run_length_current" in sdf.columns:
        X_data["run_length"] = sdf["run_length_current"]
        feature_cols.append("run_length")

    if not feature_cols:
        return None

    X = pd.DataFrame(X_data, index=sdf.index)
    y = sdf["choice_a"].values

    # Drop NaN rows (first row has NaN from shift)
    valid = X.notna().all(axis=1)
    X = X[valid].values
    y = y[valid.values]

    if len(y) < 10 or len(np.unique(y)) < 2:
        return None

    model = LogisticRegression(C=np.inf, max_iter=1000, solver="lbfgs")
    model.fit(X, y)

    probs = model.predict_proba(X)[:, 1]
    nll = log_loss(y, probs, normalize=False)

    return {
        "n_params": len(feature_cols) + 1,  # +1 for intercept
        "nll": nll,
        "accuracy": accuracy_score(y, model.predict(X)),
        "coefficients": dict(zip(feature_cols, model.coef_[0])),
    }
