"""Reinforcement learning models: Q-learning variants."""

import numpy as np
import pandas as pd
from scipy.optimize import minimize


def fit_rl_models(events_df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Fit RL models per session. Returns model comparison table."""
    n_starts = config.get("rl_models", {}).get("n_starts", 10)
    rows = []

    for sid, sdf in events_df.groupby("session_id"):
        sdf = sdf.sort_values("timestamp_ms").reset_index(drop=True)
        choices = sdf["choice_a"].values
        rewards = sdf["reward_outcome"].values

        if len(choices) < 20:
            continue

        # Basic Q-learning (alpha, beta)
        result = _fit_qlearning(choices, rewards, n_starts)
        rows.append(_rl_row(sid, "q_learning", result, len(choices)))

        # Dual learning rate (alpha_pos, alpha_neg, beta)
        result = _fit_dual_alpha(choices, rewards, n_starts)
        rows.append(_rl_row(sid, "q_dual_alpha", result, len(choices)))

        # Forgetting Q-learning (alpha, beta, forget)
        result = _fit_forgetting_q(choices, rewards, n_starts)
        rows.append(_rl_row(sid, "q_forgetting", result, len(choices)))

    return pd.DataFrame(rows)


def _rl_row(session_id, model_name, result, n_obs):
    """Create standardized row from RL fit result."""
    n_params = len(result["params"])
    nll = result["nll"]
    aic = 2 * n_params + 2 * nll
    bic = n_params * np.log(n_obs) + 2 * nll
    row = {
        "session_id": session_id,
        "model": model_name,
        "n_params": n_params,
        "nll": nll,
        "aic": aic,
        "bic": bic,
        "n_obs": n_obs,
    }
    row.update(result["params"])
    return row


def _fit_qlearning(choices: np.ndarray, rewards: np.ndarray,
                   n_starts: int) -> dict:
    """Standard Q-learning: Q(a) += alpha * (r - Q(a)), softmax choice."""
    def neg_log_lik(params):
        alpha, beta = params
        alpha = _sigmoid(alpha)
        beta = np.exp(beta)  # ensure positive
        return _qlearning_nll(choices, rewards, alpha, beta)

    best = _multi_start_optimize(neg_log_lik, n_starts, n_params=2,
                                  bounds=[(-5, 5), (-2, 5)])
    alpha = _sigmoid(best.x[0])
    beta = np.exp(best.x[1])
    return {"nll": best.fun, "params": {"alpha": alpha, "beta": beta}}


def _fit_dual_alpha(choices: np.ndarray, rewards: np.ndarray,
                    n_starts: int) -> dict:
    """Q-learning with separate learning rates for positive/negative RPE."""
    def neg_log_lik(params):
        alpha_pos, alpha_neg, beta = params
        alpha_pos = _sigmoid(alpha_pos)
        alpha_neg = _sigmoid(alpha_neg)
        beta = np.exp(beta)
        return _dual_alpha_nll(choices, rewards, alpha_pos, alpha_neg, beta)

    best = _multi_start_optimize(neg_log_lik, n_starts, n_params=3,
                                  bounds=[(-5, 5), (-5, 5), (-2, 5)])
    alpha_pos = _sigmoid(best.x[0])
    alpha_neg = _sigmoid(best.x[1])
    beta = np.exp(best.x[2])
    return {
        "nll": best.fun,
        "params": {"alpha_pos": alpha_pos, "alpha_neg": alpha_neg, "beta": beta},
    }


def _fit_forgetting_q(choices: np.ndarray, rewards: np.ndarray,
                      n_starts: int) -> dict:
    """Q-learning with forgetting: unchosen option decays toward 0.5."""
    def neg_log_lik(params):
        alpha, beta, forget = params
        alpha = _sigmoid(alpha)
        beta = np.exp(beta)
        forget = _sigmoid(forget)
        return _forgetting_q_nll(choices, rewards, alpha, beta, forget)

    best = _multi_start_optimize(neg_log_lik, n_starts, n_params=3,
                                  bounds=[(-5, 5), (-2, 5), (-5, 5)])
    alpha = _sigmoid(best.x[0])
    beta = np.exp(best.x[1])
    forget = _sigmoid(best.x[2])
    return {
        "nll": best.fun,
        "params": {"alpha": alpha, "beta": beta, "forget": forget},
    }


# ---- Core NLL functions ----

def _qlearning_nll(choices, rewards, alpha, beta):
    """Negative log-likelihood for basic Q-learning."""
    Q = np.array([0.5, 0.5])  # Q[0]=A, Q[1]=B
    nll = 0.0
    for t in range(len(choices)):
        p_a = _softmax_prob(Q[0], Q[1], beta)
        nll -= np.log(max(p_a if choices[t] == 1 else 1 - p_a, 1e-10))

        chosen = 0 if choices[t] == 1 else 1
        rpe = rewards[t] - Q[chosen]
        Q[chosen] += alpha * rpe
    return nll


def _dual_alpha_nll(choices, rewards, alpha_pos, alpha_neg, beta):
    """NLL for dual-learning-rate Q-learning."""
    Q = np.array([0.5, 0.5])
    nll = 0.0
    for t in range(len(choices)):
        p_a = _softmax_prob(Q[0], Q[1], beta)
        nll -= np.log(max(p_a if choices[t] == 1 else 1 - p_a, 1e-10))

        chosen = 0 if choices[t] == 1 else 1
        rpe = rewards[t] - Q[chosen]
        alpha = alpha_pos if rpe > 0 else alpha_neg
        Q[chosen] += alpha * rpe
    return nll


def _forgetting_q_nll(choices, rewards, alpha, beta, forget):
    """NLL for Q-learning with forgetting (unchosen decay)."""
    Q = np.array([0.5, 0.5])
    nll = 0.0
    for t in range(len(choices)):
        p_a = _softmax_prob(Q[0], Q[1], beta)
        nll -= np.log(max(p_a if choices[t] == 1 else 1 - p_a, 1e-10))

        chosen = 0 if choices[t] == 1 else 1
        unchosen = 1 - chosen
        rpe = rewards[t] - Q[chosen]
        Q[chosen] += alpha * rpe
        Q[unchosen] += forget * (0.5 - Q[unchosen])  # decay toward 0.5
    return nll


# ---- Helpers ----

def _softmax_prob(q_a, q_b, beta):
    """P(choose A) under softmax."""
    dv = beta * (q_a - q_b)
    dv = np.clip(dv, -500, 500)
    return 1.0 / (1.0 + np.exp(-dv))


def _sigmoid(x):
    """Sigmoid transform for parameters bounded in (0, 1)."""
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


def _multi_start_optimize(func, n_starts, n_params, bounds):
    """Run optimization from multiple random starting points."""
    best = None
    rng = np.random.default_rng(42)
    for _ in range(n_starts):
        x0 = rng.uniform([b[0] for b in bounds], [b[1] for b in bounds])
        try:
            result = minimize(func, x0, method="L-BFGS-B", bounds=bounds)
            if best is None or result.fun < best.fun:
                best = result
        except Exception:
            continue
    if best is None:
        # Fallback
        x0 = np.zeros(n_params)
        best = minimize(func, x0, method="L-BFGS-B", bounds=bounds)
    return best
