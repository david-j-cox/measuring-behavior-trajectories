"""Historical behavior-dynamic models of choice.

Implements four models from the 1980s–1992 behavior dynamics literature:
  - Melioration (Herrnstein & Vaughan, 1980; Vaughan, 1981)
  - Kinetic model (Myerson & Miezin, 1980; Myerson & Hale, 1988)
  - Behavioral momentum (Nevin, 1983; Nevin & Shahan, 2011)
  - Hill-climbing / momentary maximizing (Hinson & Staddon, 1983)

Each model produces both:
  - Trial-level predictions: P(choose A) per trial -> NLL, AIC, BIC
  - Trajectory-level predictions: predicted choice proportion per time bin -> RMSE, R²
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize


# ============================================================
# Public API
# ============================================================

def fit_historical_models(events_df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Fit all historical dynamics models per session."""
    n_starts = config.get("rl_models", {}).get("n_starts", 10)
    window = config.get("rolling_window_clicks", 20)
    rows = []

    for sid, sdf in events_df.groupby("session_id"):
        sdf = sdf.sort_values("timestamp_ms").reset_index(drop=True)
        choices = sdf["choice_a"].values
        rewards = sdf["reward_outcome"].values

        if len(choices) < 30:
            continue

        n_obs = len(choices)

        # Melioration (Herrnstein & Vaughan, 1980; Vaughan, 1981)
        result = _fit_melioration(choices, rewards, n_starts)
        rows.append(_model_row(sid, "melioration", result, n_obs))

        # Kinetic model (Myerson & Miezin, 1980)
        local_rates = _get_local_rates(sdf, window)
        result = _fit_kinetic(choices, local_rates, n_starts)
        rows.append(_model_row(sid, "kinetic", result, n_obs))

        # Behavioral momentum (Nevin, 1983, 1992)
        phases = sdf["phase_id"].values if "phase_id" in sdf.columns else None
        result = _fit_momentum(choices, rewards, phases, n_starts)
        rows.append(_model_row(sid, "behavioral_momentum", result, n_obs))

        # Hill-climbing (Hinson & Staddon, 1983)
        ici_s = sdf["ici_s"].values if "ici_s" in sdf.columns else None
        result = _fit_hill_climbing(choices, rewards, ici_s, n_starts)
        rows.append(_model_row(sid, "hill_climbing", result, n_obs))

    return pd.DataFrame(rows)


# ============================================================
# Melioration (Herrnstein & Vaughan, 1980; Vaughan, 1981)
# ============================================================
# Core idea: track local reinforcement rate for each option via
# exponential recency weighting. Shift choice toward the option
# with the higher local rate. Equilibrium = matching law.
#
# R_i(t) = (1 - alpha) * R_i(t-1) + alpha * reward_i(t)
# P(A) = sigmoid(beta * (R_A - R_B))

def _fit_melioration(choices, rewards, n_starts):
    """Fit melioration model: alpha (learning rate), beta (sensitivity)."""
    def neg_log_lik(params):
        alpha = _sigmoid(params[0])
        beta = np.exp(params[1])
        return _melioration_nll(choices, rewards, alpha, beta)

    best = _multi_start_optimize(neg_log_lik, n_starts, n_params=2,
                                  bounds=[(-5, 5), (-2, 5)])
    alpha = _sigmoid(best.x[0])
    beta = np.exp(best.x[1])
    return {
        "nll": best.fun,
        "n_params": 2,
        "params": {"alpha": alpha, "beta": beta},
    }


def _melioration_nll(choices, rewards, alpha, beta):
    """Negative log-likelihood for melioration."""
    R_A, R_B = 0.5, 0.5
    nll = 0.0
    for t in range(len(choices)):
        # Choice probability based on local rate difference
        p_a = _logistic(beta * (R_A - R_B))
        nll -= np.log(max(p_a if choices[t] == 1 else 1 - p_a, 1e-10))

        # Update local rate estimate for chosen option only
        if choices[t] == 1:
            R_A = (1 - alpha) * R_A + alpha * rewards[t]
        else:
            R_B = (1 - alpha) * R_B + alpha * rewards[t]
    return nll


# ============================================================
# Kinetic Model (Myerson & Miezin, 1980; Myerson & Hale, 1988)
# ============================================================
# Core idea: choice proportion P(A) evolves via a differential
# equation driven by obtained reinforcement rates.
#
# dP/dt = k * R_A * (1 - P) - k * R_B * P
#
# Equilibrium: P* = R_A / (R_A + R_B) = matching law
# Trajectory: logistic approach to equilibrium
# k = rate constant governing speed of preference change

def _fit_kinetic(choices, local_rates, n_starts):
    """Fit kinetic model: k (rate constant), beta (choice noise)."""
    def neg_log_lik(params):
        k = np.exp(params[0])  # ensure positive
        beta = np.exp(params[1])
        return _kinetic_nll(choices, local_rates, k, beta)

    best = _multi_start_optimize(neg_log_lik, n_starts, n_params=2,
                                  bounds=[(-6, 2), (-2, 5)])
    k = np.exp(best.x[0])
    beta = np.exp(best.x[1])
    return {
        "nll": best.fun,
        "n_params": 2,
        "params": {"k": k, "beta": beta},
    }


def _kinetic_nll(choices, local_rates, k, beta):
    """Negative log-likelihood for kinetic model.

    Uses locally estimated reinforcement rates to drive the
    differential equation, discretized per trial.
    """
    P = 0.5  # initial choice proportion
    nll = 0.0
    R_A_vals, R_B_vals = local_rates

    for t in range(len(choices)):
        # Choice probability: kinetic P + noise via softmax
        p_a = _logistic(beta * (P - 0.5))
        nll -= np.log(max(p_a if choices[t] == 1 else 1 - p_a, 1e-10))

        # Update P via discretized kinetic equation
        R_A = R_A_vals[t] if t < len(R_A_vals) else 0.5
        R_B = R_B_vals[t] if t < len(R_B_vals) else 0.5
        dP = k * R_A * (1 - P) - k * R_B * P
        P = np.clip(P + dP, 0.001, 0.999)

    return nll


# ============================================================
# Behavioral Momentum (Nevin, 1983; Nevin & Shahan, 2011)
# ============================================================
# Core idea: behavior accumulates "mass" proportional to
# reinforcement rate. When contingencies shift, the prior
# preference resists change in proportion to r^0.5.
#
# B_t / B_0 = 10^((-t * (c + d*r)) / r^0.5)
#
# For choice between two options after a regime shift:
# - Old preference decays (resisted by prior reinforcement)
# - New preference grows (with its own momentum)
# - Asymmetric transitions emerge because reversals must
#   overcome existing momentum

def _fit_momentum(choices, rewards, phases, n_starts):
    """Fit behavioral momentum: c, d (disruption params), beta (noise).

    c = baseline disruption sensitivity
    d = reinforcement-dependent disruption scaling
    beta = choice sensitivity
    """
    if phases is None:
        # Can't fit momentum without phase information
        return {
            "nll": np.inf,
            "n_params": 3,
            "params": {"c": np.nan, "d": np.nan, "beta": np.nan},
        }

    def neg_log_lik(params):
        c = np.exp(params[0])
        d = np.exp(params[1])
        beta = np.exp(params[2])
        return _momentum_nll(choices, rewards, phases, c, d, beta)

    best = _multi_start_optimize(neg_log_lik, n_starts, n_params=3,
                                  bounds=[(-4, 2), (-6, 0), (-2, 5)])
    c = np.exp(best.x[0])
    d = np.exp(best.x[1])
    beta = np.exp(best.x[2])
    return {
        "nll": best.fun,
        "n_params": 3,
        "params": {"c": c, "d": d, "beta": beta},
    }


def _momentum_nll(choices, rewards, phases, c, d, beta):
    """Negative log-likelihood for behavioral momentum model.

    Tracks accumulated reinforcement per option within each phase.
    At phase transitions, uses momentum equations to predict
    resistance to change.
    """
    nll = 0.0

    # Track reinforcement history per option
    r_A_accum = 0.0  # accumulated reward rate for A
    r_B_accum = 0.0
    n_A = 0
    n_B = 0
    current_phase = phases[0]
    phase_start_t = 0

    # Strength of preference toward A (positive = prefer A)
    strength_A = 0.0
    strength_B = 0.0

    for t in range(len(choices)):
        # Detect phase transition
        if phases[t] != current_phase:
            # Compute reinforcement rates from prior phase
            r_A = r_A_accum / max(n_A, 1)
            r_B = r_B_accum / max(n_B, 1)

            # Prior strengths become resistant to change
            # Momentum: resistance proportional to sqrt of reinforcement rate
            strength_A = r_A
            strength_B = r_B

            # Reset accumulators for new phase
            r_A_accum = 0.0
            r_B_accum = 0.0
            n_A = 0
            n_B = 0
            current_phase = phases[t]
            phase_start_t = t

        # Time since phase start (for decay calculation)
        t_in_phase = t - phase_start_t + 1

        # Old preference decays with momentum-dependent resistance
        decay_A = 10 ** ((-t_in_phase * (c + d * strength_A)) /
                         max(strength_A ** 0.5, 0.01))
        decay_B = 10 ** ((-t_in_phase * (c + d * strength_B)) /
                         max(strength_B ** 0.5, 0.01))

        # Current reinforcement rates in this phase
        curr_r_A = r_A_accum / max(n_A, 1) if n_A > 0 else 0.5
        curr_r_B = r_B_accum / max(n_B, 1) if n_B > 0 else 0.5

        # Net preference = current evidence + decaying prior
        prefer_A = curr_r_A + strength_A * decay_A
        prefer_B = curr_r_B + strength_B * decay_B
        diff = prefer_A - prefer_B

        p_a = _logistic(beta * diff)
        nll -= np.log(max(p_a if choices[t] == 1 else 1 - p_a, 1e-10))

        # Accumulate reinforcement
        if choices[t] == 1:
            r_A_accum += rewards[t]
            n_A += 1
        else:
            r_B_accum += rewards[t]
            n_B += 1

    return nll


# ============================================================
# Hill-Climbing / Momentary Maximizing (Hinson & Staddon, 1983)
# ============================================================
# Core idea: on each trial, choose the option with the higher
# current reinforcement probability, estimated from time since
# last response to each option.
#
# P(reward | option i) = 1 - exp(-lambda_i * t_i)
#
# where t_i = time since last response to option i.
# With optional recency parameter A weighting recent reinforcement.
#
# Generalized: choose B when t_A + A/T_A > t_B + A/T_B
# where T_i = time since last reinforcer on option i.

def _fit_hill_climbing(choices, rewards, ici_s, n_starts):
    """Fit hill-climbing: A (recency weight), beta (noise)."""
    if ici_s is None:
        return {
            "nll": np.inf,
            "n_params": 2,
            "params": {"A_recency": np.nan, "beta": np.nan},
        }

    def neg_log_lik(params):
        A = np.exp(params[0])  # recency parameter, positive
        beta = np.exp(params[1])
        return _hill_climbing_nll(choices, rewards, ici_s, A, beta)

    best = _multi_start_optimize(neg_log_lik, n_starts, n_params=2,
                                  bounds=[(-4, 4), (-2, 5)])
    A = np.exp(best.x[0])
    beta = np.exp(best.x[1])
    return {
        "nll": best.fun,
        "n_params": 2,
        "params": {"A_recency": A, "beta": beta},
    }


def _hill_climbing_nll(choices, rewards, ici_s, A, beta):
    """Negative log-likelihood for hill-climbing model.

    Tracks time since last response and time since last reinforcer
    for each option. Computes momentary reinforcement probability
    and selects the option with higher expected payoff.
    """
    nll = 0.0

    # Time since last response to each option (in seconds)
    t_A = 1.0  # initialize to 1 second
    t_B = 1.0

    # Time since last reinforcer on each option
    T_A = 10.0  # initialize to large value (no recent reinforcer)
    T_B = 10.0

    for t in range(len(choices)):
        # Momentary value: time since last response + recency bonus
        # Higher value = more likely to be reinforced (recovery time)
        val_A = t_A + A / max(T_A, 0.01)
        val_B = t_B + A / max(T_B, 0.01)

        # Choice probability via softmax on momentary values
        p_a = _logistic(beta * (val_A - val_B))
        nll -= np.log(max(p_a if choices[t] == 1 else 1 - p_a, 1e-10))

        # Update time-since-last-response
        dt = ici_s[t] if t < len(ici_s) and not np.isnan(ici_s[t]) else 0.4
        if choices[t] == 1:
            t_A = dt  # just responded to A
            t_B += dt  # B continues to recover
        else:
            t_B = dt
            t_A += dt

        # Update time-since-last-reinforcer
        T_A += dt
        T_B += dt
        if choices[t] == 1 and rewards[t] == 1:
            T_A = dt
        elif choices[t] == 0 and rewards[t] == 1:
            T_B = dt

    return nll


# ============================================================
# Trajectory-level predictions
# ============================================================

def predict_trajectories(events_df: pd.DataFrame, config: dict,
                         fit_results: pd.DataFrame) -> dict:
    """Generate trajectory-level predictions for all historical models.

    Returns dict of {model_name: DataFrame with session_id, bin, observed, predicted}.
    """
    window = config.get("rolling_window_clicks", 20)
    all_predictions = {}

    for model_name in ["melioration", "kinetic", "behavioral_momentum", "hill_climbing"]:
        model_fits = fit_results[fit_results["model"] == model_name]
        if len(model_fits) == 0:
            continue

        pred_rows = []
        for _, fit_row in model_fits.iterrows():
            sid = fit_row["session_id"]
            sdf = events_df[events_df["session_id"] == sid].sort_values(
                "timestamp_ms").reset_index(drop=True)
            if len(sdf) < window:
                continue

            choices = sdf["choice_a"].values
            rewards = sdf["reward_outcome"].values

            # Generate trial-level P(A) predictions
            if model_name == "melioration":
                p_a_seq = _melioration_predict(
                    choices, rewards,
                    fit_row.get("alpha", 0.1), fit_row.get("beta", 5.0))
            elif model_name == "kinetic":
                local_rates = _get_local_rates(sdf, window)
                p_a_seq = _kinetic_predict(
                    choices, local_rates,
                    fit_row.get("k", 0.01), fit_row.get("beta", 5.0))
            elif model_name == "behavioral_momentum":
                phases = sdf["phase_id"].values if "phase_id" in sdf.columns else None
                p_a_seq = _momentum_predict(
                    choices, rewards, phases,
                    fit_row.get("c", 0.1), fit_row.get("d", 0.01),
                    fit_row.get("beta", 5.0))
            elif model_name == "hill_climbing":
                ici_s = sdf["ici_s"].values if "ici_s" in sdf.columns else None
                p_a_seq = _hill_climbing_predict(
                    choices, rewards, ici_s,
                    fit_row.get("A_recency", 1.0), fit_row.get("beta", 5.0))
            else:
                continue

            # Bin into non-overlapping windows
            n_bins = len(choices) // window
            for b in range(n_bins):
                start = b * window
                end = start + window
                obs = np.mean(choices[start:end])
                pred = np.mean(p_a_seq[start:end])
                pred_rows.append({
                    "session_id": sid,
                    "bin": b,
                    "bin_start": start,
                    "observed": obs,
                    "predicted": pred,
                })

        all_predictions[model_name] = pd.DataFrame(pred_rows)

    return all_predictions


# ---- Prediction generators (same logic as NLL but return P(A) sequence) ----

def _melioration_predict(choices, rewards, alpha, beta):
    R_A, R_B = 0.5, 0.5
    p_a_seq = np.zeros(len(choices))
    for t in range(len(choices)):
        p_a_seq[t] = _logistic(beta * (R_A - R_B))
        if choices[t] == 1:
            R_A = (1 - alpha) * R_A + alpha * rewards[t]
        else:
            R_B = (1 - alpha) * R_B + alpha * rewards[t]
    return p_a_seq


def _kinetic_predict(choices, local_rates, k, beta):
    P = 0.5
    R_A_vals, R_B_vals = local_rates
    p_a_seq = np.zeros(len(choices))
    for t in range(len(choices)):
        p_a_seq[t] = _logistic(beta * (P - 0.5))
        R_A = R_A_vals[t] if t < len(R_A_vals) else 0.5
        R_B = R_B_vals[t] if t < len(R_B_vals) else 0.5
        dP = k * R_A * (1 - P) - k * R_B * P
        P = np.clip(P + dP, 0.001, 0.999)
    return p_a_seq


def _momentum_predict(choices, rewards, phases, c, d, beta):
    p_a_seq = np.zeros(len(choices))
    if phases is None:
        p_a_seq[:] = 0.5
        return p_a_seq

    r_A_accum = 0.0
    r_B_accum = 0.0
    n_A = 0
    n_B = 0
    current_phase = phases[0]
    phase_start_t = 0
    strength_A = 0.0
    strength_B = 0.0

    for t in range(len(choices)):
        if phases[t] != current_phase:
            r_A = r_A_accum / max(n_A, 1)
            r_B = r_B_accum / max(n_B, 1)
            strength_A = r_A
            strength_B = r_B
            r_A_accum = 0.0
            r_B_accum = 0.0
            n_A = 0
            n_B = 0
            current_phase = phases[t]
            phase_start_t = t

        t_in_phase = t - phase_start_t + 1
        decay_A = 10 ** ((-t_in_phase * (c + d * strength_A)) /
                         max(strength_A ** 0.5, 0.01))
        decay_B = 10 ** ((-t_in_phase * (c + d * strength_B)) /
                         max(strength_B ** 0.5, 0.01))
        curr_r_A = r_A_accum / max(n_A, 1) if n_A > 0 else 0.5
        curr_r_B = r_B_accum / max(n_B, 1) if n_B > 0 else 0.5
        prefer_A = curr_r_A + strength_A * decay_A
        prefer_B = curr_r_B + strength_B * decay_B
        diff = prefer_A - prefer_B
        p_a_seq[t] = _logistic(beta * diff)

        if choices[t] == 1:
            r_A_accum += rewards[t]
            n_A += 1
        else:
            r_B_accum += rewards[t]
            n_B += 1

    return p_a_seq


def _hill_climbing_predict(choices, rewards, ici_s, A, beta):
    p_a_seq = np.zeros(len(choices))
    t_A = 1.0
    t_B = 1.0
    T_A = 10.0
    T_B = 10.0

    for t in range(len(choices)):
        val_A = t_A + A / max(T_A, 0.01)
        val_B = t_B + A / max(T_B, 0.01)
        p_a_seq[t] = _logistic(beta * (val_A - val_B))

        dt = ici_s[t] if (ici_s is not None and t < len(ici_s) and
                          not np.isnan(ici_s[t])) else 0.4
        if choices[t] == 1:
            t_A = dt
            t_B += dt
        else:
            t_B = dt
            t_A += dt
        T_A += dt
        T_B += dt
        if choices[t] == 1 and rewards[t] == 1:
            T_A = dt
        elif choices[t] == 0 and rewards[t] == 1:
            T_B = dt

    return p_a_seq


# ============================================================
# Helpers
# ============================================================

def _get_local_rates(sdf, window):
    """Extract local reinforcement rates for options A and B."""
    if "local_reward_rate_a" in sdf.columns and "local_reward_rate_b" in sdf.columns:
        R_A = sdf["local_reward_rate_a"].fillna(0.5).values
        R_B = sdf["local_reward_rate_b"].fillna(0.5).values
    else:
        # Compute from raw data using rolling window
        choices = sdf["choice_a"].values
        rewards = sdf["reward_outcome"].values
        R_A = np.full(len(choices), 0.5)
        R_B = np.full(len(choices), 0.5)
        for t in range(window, len(choices)):
            win = slice(t - window, t)
            c = choices[win]
            r = rewards[win]
            n_a = c.sum()
            n_b = window - n_a
            R_A[t] = (c * r).sum() / max(n_a, 1)
            R_B[t] = ((1 - c) * r).sum() / max(n_b, 1)
    return R_A, R_B


def _model_row(session_id, model_name, result, n_obs):
    """Create standardized model comparison row."""
    n_params = result["n_params"]
    nll = result["nll"]
    row = {
        "session_id": session_id,
        "model": model_name,
        "n_params": n_params,
        "nll": nll,
        "aic": 2 * n_params + 2 * nll if np.isfinite(nll) else np.nan,
        "bic": (n_params * np.log(n_obs) + 2 * nll
                if np.isfinite(nll) else np.nan),
        "n_obs": n_obs,
    }
    row.update(result["params"])
    return row


def _logistic(x):
    """Logistic sigmoid, numerically stable."""
    x = np.clip(x, -500, 500)
    return 1.0 / (1.0 + np.exp(-x))


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
        x0 = np.zeros(n_params)
        best = minimize(func, x0, method="L-BFGS-B", bounds=bounds)
    return best
