"""Reinforcement learning models: Q-learning variants."""

import numpy as np
import pandas as pd
from scipy.optimize import minimize


def fit_rl_models(events_df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Fit RL models per session. Returns model comparison table."""
    n_starts = config.get("rl_models", {}).get("n_starts", 10)
    phase_boundaries = config.get("phase_boundaries", [])
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

        # Dynamic learning rate (alpha_base, alpha_gain, beta, decay)
        result = _fit_dynamic_alpha(choices, rewards, n_starts)
        rows.append(_rl_row(sid, "q_dynamic_alpha", result, len(choices)))

        # Phase-aware Q-learning (separate alpha per phase)
        if "phase_id" in sdf.columns and len(phase_boundaries) > 0:
            phases = sdf["phase_id"].values
            result = _fit_phase_aware_q(choices, rewards, phases,
                                        phase_boundaries, n_starts)
            rows.append(_rl_row(sid, "q_phase_aware", result, len(choices)))

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


def _fit_dynamic_alpha(choices: np.ndarray, rewards: np.ndarray,
                       n_starts: int) -> dict:
    """
    Q-learning with dynamic learning rate that increases after
    prediction errors (surprise-modulated learning).

    alpha_t = alpha_base + alpha_gain * |RPE_t-1|
    This captures the idea that large surprises trigger faster updating,
    which is adaptive in non-stationary environments.
    """
    def neg_log_lik(params):
        alpha_base, alpha_gain, beta, decay = params
        alpha_base = _sigmoid(alpha_base)
        alpha_gain = _sigmoid(alpha_gain)
        beta = np.exp(beta)
        decay = _sigmoid(decay)
        return _dynamic_alpha_nll(choices, rewards, alpha_base, alpha_gain,
                                  beta, decay)

    best = _multi_start_optimize(neg_log_lik, n_starts, n_params=4,
                                  bounds=[(-5, 5), (-5, 5), (-2, 5), (-5, 5)])
    alpha_base = _sigmoid(best.x[0])
    alpha_gain = _sigmoid(best.x[1])
    beta = np.exp(best.x[2])
    decay = _sigmoid(best.x[3])
    return {
        "nll": best.fun,
        "params": {
            "alpha_base": alpha_base,
            "alpha_gain": alpha_gain,
            "beta": beta,
            "decay": decay,
        },
    }


def _fit_phase_aware_q(choices: np.ndarray, rewards: np.ndarray,
                       phases: np.ndarray, phase_boundaries: list,
                       n_starts: int) -> dict:
    """
    Q-learning with separate learning rates per phase but shared beta.

    This allows the model to capture different adaptation speeds across
    the four environmental regimes, directly testing whether participants
    adjust their learning rate to environmental volatility.
    """
    n_phases = len(phase_boundaries)
    # Parameters: alpha_1, alpha_2, ..., alpha_N, beta
    n_params = n_phases + 1

    phase_ids = sorted(set(p["id"] for p in phase_boundaries))
    phase_map = {pid: i for i, pid in enumerate(phase_ids)}

    def neg_log_lik(params):
        alphas_raw = params[:n_phases]
        beta_raw = params[n_phases]
        alphas = np.array([_sigmoid(a) for a in alphas_raw])
        beta = np.exp(beta_raw)
        return _phase_aware_nll(choices, rewards, phases, alphas,
                                beta, phase_map)

    bounds = [(-5, 5)] * n_phases + [(-2, 5)]
    best = _multi_start_optimize(neg_log_lik, n_starts, n_params=n_params,
                                  bounds=bounds)

    params_dict = {"beta": np.exp(best.x[n_phases])}
    for i, pid in enumerate(phase_ids):
        params_dict[f"alpha_phase{pid}"] = _sigmoid(best.x[i])

    return {"nll": best.fun, "params": params_dict}


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


def _dynamic_alpha_nll(choices, rewards, alpha_base, alpha_gain, beta, decay):
    """NLL for Q-learning with surprise-modulated learning rate."""
    Q = np.array([0.5, 0.5])
    nll = 0.0
    prev_unsigned_rpe = 0.0

    for t in range(len(choices)):
        p_a = _softmax_prob(Q[0], Q[1], beta)
        nll -= np.log(max(p_a if choices[t] == 1 else 1 - p_a, 1e-10))

        chosen = 0 if choices[t] == 1 else 1
        rpe = rewards[t] - Q[chosen]

        # Dynamic alpha: boost after large surprises
        alpha_t = min(1.0, alpha_base + alpha_gain * prev_unsigned_rpe)
        Q[chosen] += alpha_t * rpe

        # Track surprise with exponential decay
        prev_unsigned_rpe = (decay * prev_unsigned_rpe +
                             (1 - decay) * abs(rpe))
    return nll


def _phase_aware_nll(choices, rewards, phases, alphas, beta, phase_map):
    """NLL for Q-learning with phase-specific learning rates."""
    Q = np.array([0.5, 0.5])
    nll = 0.0

    for t in range(len(choices)):
        p_a = _softmax_prob(Q[0], Q[1], beta)
        nll -= np.log(max(p_a if choices[t] == 1 else 1 - p_a, 1e-10))

        chosen = 0 if choices[t] == 1 else 1
        rpe = rewards[t] - Q[chosen]

        # Use phase-specific alpha
        phase_idx = phase_map.get(phases[t], 0)
        alpha = alphas[phase_idx]
        Q[chosen] += alpha * rpe
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
