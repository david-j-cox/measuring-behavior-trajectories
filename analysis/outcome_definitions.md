# Outcome Variable Definitions

This document defines all primary, secondary, and exploratory outcome variables computed by the analysis pipeline. Each entry includes an operational definition, formula (where applicable), units, interpretation guidance, and threats to validity.

---

## Classification Key

| Label | Meaning |
|-------|---------|
| **Primary** | Pre-registered confirmatory measures central to the research questions. |
| **Secondary** | Supportive measures that contextualize the primary outcomes. |
| **Exploratory** | Post-hoc or novel measures for hypothesis generation; interpret with caution. |

---

## 1. Choice Allocation

### 1.1 Choice Proportion — P(A)

| Field | Value |
|-------|-------|
| **Variable name** | `choice_prop_a` (session-level), `rolling_choice_prop_a_clicks` (event-level) |
| **Classification** | Primary |
| **Definition** | The proportion of clicks allocated to Option A within a given window or session. |
| **Formula** | $P(A) = \frac{\sum_{t} \mathbf{1}[\text{choice}_t = A]}{N}$ |
| **Units** | Proportion, range [0, 1] |
| **Interpretation** | Values near 0.5 indicate indifference; deviations indicate preference. In phases where one option is objectively better, departure from the advantaged option indicates suboptimal tracking. |
| **Threats** | Sensitive to session length; short sessions yield noisy estimates. Rolling estimates depend on window size (default: 20 clicks). Does not distinguish between stable preference and rapid oscillation that averages to 0.5. |

### 1.2 Choice Entropy

| Field | Value |
|-------|-------|
| **Variable name** | `choice_entropy` |
| **Classification** | Secondary |
| **Definition** | Shannon entropy of the binary choice distribution within a session. |
| **Formula** | $H = -[P(A) \log_2 P(A) + P(B) \log_2 P(B)]$ |
| **Units** | Bits, range [0, 1] |
| **Interpretation** | 1.0 = maximum uncertainty (50/50 allocation); 0.0 = exclusive preference for one option. Low entropy may indicate strong preference or perseveration. |
| **Threats** | A session-level summary; does not capture within-session dynamics. Two sessions with identical entropy can have very different temporal patterns. |

### 1.3 Choice Bias (Recent Window)

| Field | Value |
|-------|-------|
| **Variable name** | `choice_bias_recent` |
| **Classification** | Exploratory |
| **Definition** | Normalized difference between recent counts of A and B choices within a rolling window. |
| **Formula** | $\text{bias} = \frac{n_A - n_B}{n_A + n_B}$ over the most recent $w$ clicks |
| **Units** | Dimensionless, range [-1, 1] |
| **Interpretation** | +1 = all recent choices are A; −1 = all B; 0 = balanced. Tracks moment-to-moment allocation shifts. |
| **Threats** | Window size (default 20 clicks) is arbitrary. Brief bursts of one option inflate the metric transiently. |

---

## 2. Reward and Optimality

### 2.1 Overall Reward Rate

| Field | Value |
|-------|-------|
| **Variable name** | `overall_reward_rate` (session), `rolling_reward_rate_clicks` (event) |
| **Classification** | Primary |
| **Definition** | Proportion of clicks that produced a reward (point delivery). |
| **Formula** | $\bar{r} = \frac{\sum_t r_t}{N}$, where $r_t \in \{0, 1\}$ |
| **Units** | Proportion, range [0, 1] |
| **Interpretation** | Higher values indicate more successful foraging. Because reward probability depends on hidden latent values, reward rate reflects both choice quality and stochastic variability. |
| **Threats** | Confounded with the environment's generosity (phase-dependent base rates). Phase-specific reward rates are more informative than the overall rate. |

### 2.2 Proportion Optimal

| Field | Value |
|-------|-------|
| **Variable name** | `proportion_optimal`, `choice_optimal` (event-level binary) |
| **Classification** | Primary |
| **Definition** | Proportion of choices that selected the option with the higher latent value at the moment of the click. |
| **Formula** | $\text{prop\_optimal} = \frac{1}{N}\sum_t \mathbf{1}[\text{choice}_t = \arg\max(V_A^{(t)}, V_B^{(t)})]$ |
| **Units** | Proportion, range [0, 1] |
| **Interpretation** | 1.0 = perfect tracking of the better option; 0.5 = chance. Because latent values fluctuate continuously, this captures how well participants track the hidden state. |
| **Threats** | Requires access to latent values (logged by the task). When latent values are very close, "optimal" vs "suboptimal" is a distinction without practical consequence. Consider gating on trials where the latent advantage exceeds a minimum threshold. |

### 2.3 Regret

| Field | Value |
|-------|-------|
| **Variable name** | `regret_proxy` (per-click), `cumulative_regret` (running sum), `mean_regret` (session) |
| **Classification** | Secondary |
| **Definition** | The difference in latent value between the best available option and the chosen option. |
| **Formula** | $\text{regret}_t = \max(V_A^{(t)}, V_B^{(t)}) - V_{\text{chosen}}^{(t)}$ |
| **Units** | Latent-value units (probability scale), range [0, 1] |
| **Interpretation** | Higher regret indicates larger missed opportunity. Cumulative regret grows over time; its slope indicates learning efficiency. |
| **Threats** | Regret is computed from latent values the participant cannot observe; it measures deviation from an omniscient ideal, not from a realistic decision standard. |

### 2.4 Local Reward Rate by Option

| Field | Value |
|-------|-------|
| **Variable name** | `local_reward_rate_a`, `local_reward_rate_b` |
| **Classification** | Secondary |
| **Definition** | Reward rate for each option computed within a rolling window of recent clicks. |
| **Formula** | $\hat{r}_A = \frac{\text{rewards from A in window}}{\text{A-choices in window}}$ (analogous for B) |
| **Units** | Proportion, range [0, 1]; undefined (NaN) if the option was not chosen in the window |
| **Interpretation** | Estimates the participant's recent experienced reward probability for each option. Divergence between options should drive switching. |
| **Threats** | Undefined when an option hasn't been sampled recently. Small denominators yield noisy estimates. |

---

## 3. Switching and Persistence

### 3.1 Switch Rate

| Field | Value |
|-------|-------|
| **Variable name** | `overall_switch_rate` (session), `rolling_switch_rate_clicks` (event), `switch_flag_verified` (event binary) |
| **Classification** | Primary |
| **Definition** | Proportion of transitions in which the participant changed from one option to the other. |
| **Formula** | $\text{switch\_rate} = \frac{\sum_{t=2}^{N} \mathbf{1}[\text{choice}_t \neq \text{choice}_{t-1}]}{N - 1}$ |
| **Units** | Proportion, range [0, 1] |
| **Interpretation** | High switch rate indicates exploratory or volatile behavior; low switch rate indicates persistence or exploitation. In dynamic environments, intermediate switching is adaptive. |
| **Threats** | The first click of a session has no defined switch status. Very high switch rates (alternation) can be a perseverative motor pattern rather than deliberate exploration. |

### 3.2 Run Length

| Field | Value |
|-------|-------|
| **Variable name** | `run_length_current` (event), `mean_run_length`, `median_run_length` (session) |
| **Classification** | Primary |
| **Definition** | Number of consecutive same-option choices ending at (and including) the current click. |
| **Formula** | $\text{run}_t = \begin{cases} 1 & \text{if } t = 1 \text{ or } c_t \neq c_{t-1} \\ \text{run}_{t-1} + 1 & \text{if } c_t = c_{t-1} \end{cases}$ |
| **Units** | Count (clicks), range [1, N] |
| **Interpretation** | Run length distributions characterize bout structure. Long mean runs indicate exploitation; short runs indicate high switching. The distribution shape (geometric, heavy-tailed) is informative about the generative process. |
| **Threats** | Mean run length is sensitive to outlier bouts. Median is more robust. Run length is mechanically related to switch rate: $E[\text{run}] \approx 1 / \text{switch\_rate}$. |

### 3.3 Previous Run Length

| Field | Value |
|-------|-------|
| **Variable name** | `run_length_previous` |
| **Classification** | Exploratory |
| **Definition** | The length of the run that ended at the most recent switch point. |
| **Units** | Count (clicks) |
| **Interpretation** | Allows analysis of what triggers a switch — do participants leave after short or long runs? Paired with reward history, this reveals giving-up thresholds. |
| **Threats** | Zero at session start (before the first switch). |

---

## 4. Win-Stay / Lose-Shift

### 4.1 Win-Stay Rate

| Field | Value |
|-------|-------|
| **Variable name** | `win_stay_rate` (session), `win_stay` (event binary, NaN on lose trials) |
| **Classification** | Primary |
| **Definition** | Probability of repeating the same choice after a rewarded trial. |
| **Formula** | $P(\text{stay} \mid \text{win}) = \frac{\sum_t \mathbf{1}[c_t = c_{t-1} \wedge r_{t-1} = 1]}{\sum_t \mathbf{1}[r_{t-1} = 1]}$ |
| **Units** | Proportion, range [0, 1] |
| **Interpretation** | High win-stay indicates reward-driven reinforcement / positive recency. In depleting environments, very high win-stay can be maladaptive because rewards deplete the current option. |
| **Threats** | Defined only for trials following a win; sample size varies by participant. Conflates true reward sensitivity with general persistence tendency. |

### 4.2 Lose-Shift Rate

| Field | Value |
|-------|-------|
| **Variable name** | `lose_shift_rate` (session), `lose_shift` (event binary, NaN on win trials) |
| **Classification** | Primary |
| **Definition** | Probability of switching options after a non-rewarded trial. |
| **Formula** | $P(\text{shift} \mid \text{lose}) = \frac{\sum_t \mathbf{1}[c_t \neq c_{t-1} \wedge r_{t-1} = 0]}{\sum_t \mathbf{1}[r_{t-1} = 0]}$ |
| **Units** | Proportion, range [0, 1] |
| **Interpretation** | High lose-shift indicates sensitivity to negative feedback. Together with win-stay, characterizes the participant's simple reactive strategy. |
| **Threats** | Same as win-stay. Additionally, in environments with low base reward rates, most trials are losses, so lose-shift is estimated more precisely than win-stay. |

---

## 5. Temporal Variables

### 5.1 Inter-Click Interval (ICI)

| Field | Value |
|-------|-------|
| **Variable name** | `ici_s` (event), `mean_ici_s`, `median_ici_s` (session) |
| **Classification** | Secondary |
| **Definition** | Time elapsed between consecutive clicks within a session. |
| **Formula** | $\text{ICI}_t = \frac{\text{timestamp}_{t} - \text{timestamp}_{t-1}}{1000}$ |
| **Units** | Seconds |
| **Interpretation** | Reflects response vigor, engagement, and deliberation. Sudden increases may indicate disengagement, distraction, or deliberation at decision points. |
| **Threats** | Susceptible to exogenous pauses (bathroom breaks, distractions). Outlier ICIs (> 10 s) may reflect inattention rather than task-related processing. The 200 ms cooldown imposes a floor. |

### 5.2 Time Since Last Reward

| Field | Value |
|-------|-------|
| **Variable name** | `time_since_last_reward_s` |
| **Classification** | Exploratory |
| **Definition** | Elapsed time (in seconds) since the most recent rewarded click. |
| **Units** | Seconds |
| **Interpretation** | Longer droughts may trigger switching. Correlating this with switch probability reveals giving-up time thresholds. |
| **Threats** | NaN before the first reward. Confounded with ICI — slow responders accumulate time without it reflecting the reward landscape. |

### 5.3 Time Since Last Switch

| Field | Value |
|-------|-------|
| **Variable name** | `time_since_last_switch_s` |
| **Classification** | Exploratory |
| **Definition** | Elapsed time since the participant last changed options. |
| **Units** | Seconds |
| **Interpretation** | Complements run length (which is count-based) with a temporal perspective. Longer dwell times on an option may indicate stronger commitment. |
| **Threats** | NaN before the first switch. |

### 5.4 Session Duration

| Field | Value |
|-------|-------|
| **Variable name** | `session_duration_s` |
| **Classification** | Secondary |
| **Definition** | Total elapsed time from first to last click in the session. |
| **Formula** | $\text{duration} = \frac{\max(\text{timestamp\_ms}) }{1000}$ |
| **Units** | Seconds |
| **Interpretation** | Sessions are designed to be 360 s (6 minutes). Shorter durations indicate early termination or exclusion. |
| **Threats** | Does not account for pre-task time (consent, instructions, practice). |

---

## 6. Phase-Level Variables

### 6.1 Phase-Specific Choice Proportion

| Field | Value |
|-------|-------|
| **Variable name** | `phase{N}_choice_prop_a` |
| **Classification** | Primary |
| **Definition** | P(A) computed within each of the 4 experimental phases. |
| **Units** | Proportion |
| **Interpretation** | Phase 1 (symmetric): expect ~0.5. Phase 2 (A advantage): expect > 0.5 if tracking. Phase 3 (B advantage): expect < 0.5 if tracking. Phase 4 (scarcity + pulses): depends on pulse-tracking ability. |
| **Threats** | Phase boundaries are defined by time, not by click count, so the number of observations per phase varies across participants. |

### 6.2 Phase-Specific Reward Rate

| Field | Value |
|-------|-------|
| **Variable name** | `phase{N}_reward_rate` |
| **Classification** | Secondary |
| **Definition** | Reward rate computed within each phase. |
| **Units** | Proportion |
| **Interpretation** | Reflects how well participants harvest rewards given that phase's environment. |

### 6.3 Adaptation Lag

| Field | Value |
|-------|-------|
| **Variable name** | `adaptation_lag_phase{N}_s` |
| **Classification** | Primary |
| **Definition** | Time (seconds) from the onset of a new phase until the participant's rolling choice proportion first crosses an adaptation threshold toward the newly advantaged option. |
| **Formula** | For Phase 2: time until $P_{\text{rolling}}(A) > 0.6$. For Phase 3: time until $P_{\text{rolling}}(A) < 0.4$. |
| **Units** | Seconds; NaN if the participant never adapts within the phase. |
| **Interpretation** | Shorter lags indicate faster behavioral adaptation to environmental change. NaN values indicate failure to adapt. |
| **Threats** | Threshold (0.6) is arbitrary — sensitivity analyses should vary it. Rolling window size affects when the criterion is met. Participants who happen to be near the threshold at phase onset will show artificially short lags. |

---

## 7. Perturbation Response Variables

### 7.1 Pulse-Triggered Choice Shift

| Field | Value |
|-------|-------|
| **Variable name** | Derived from pulse-triggered averaging (PTA); not a single stored column |
| **Classification** | Exploratory |
| **Definition** | Change in P(A) in the seconds following a bonus pulse onset, relative to the pre-pulse baseline. |
| **Units** | Proportion change |
| **Interpretation** | A-pulses should increase P(A); B-pulses should decrease P(A) — if participants detect and respond to the brief value boost. The magnitude and latency of the shift characterize perturbation sensitivity. |
| **Threats** | Pulses are brief (3 s) and occur only in Phase 4. Low click rates during the pulse window reduce statistical power. Individual pulses yield very few observations; averaging across pulses and participants is required. |

---

## 8. Dynamical / Trajectory Variables

### 8.1 Choice Autocorrelation

| Field | Value |
|-------|-------|
| **Variable name** | Computed per session at multiple lags; not a single stored column |
| **Classification** | Exploratory |
| **Definition** | Autocorrelation of the binary choice sequence at lag $k$. |
| **Formula** | $\rho(k) = \frac{\text{Cov}(c_t, c_{t+k})}{\text{Var}(c_t)}$ |
| **Units** | Correlation coefficient, range [-1, 1] |
| **Interpretation** | Positive autocorrelation at short lags indicates persistence (runs); negative autocorrelation indicates alternation. The decay rate of the ACF characterizes the timescale of behavioral momentum. |
| **Threats** | Non-stationarity (phase changes) inflates autocorrelation at long lags. Consider computing within-phase ACFs. |

### 8.2 Hysteresis Index

| Field | Value |
|-------|-------|
| **Variable name** | Derived from state-space trajectory analysis |
| **Classification** | Exploratory |
| **Definition** | Whether the mapping from latent advantage ($V_A - V_B$) to choice proportion differs depending on the direction of environmental change (Phase 2 vs Phase 3). |
| **Interpretation** | If the choice-vs-advantage curve follows a different path when advantage is increasing vs decreasing, this indicates path dependence / hysteresis. Larger enclosed area in the hysteresis loop = stronger history dependence. |
| **Threats** | Requires sufficient data in both phases. Binning the advantage axis introduces smoothing artifacts. Asymmetric phase durations or click counts can create apparent hysteresis. |

### 8.3 Recurrence

| Field | Value |
|-------|-------|
| **Variable name** | Derived from recurrence plot analysis |
| **Classification** | Exploratory |
| **Definition** | Whether the behavioral state (rolling choice proportion) at time $t$ revisits states previously occupied at time $s$. |
| **Interpretation** | Diagonal structures in the recurrence plot indicate deterministic dynamics; uniform filling indicates stochastic behavior. |
| **Threats** | Threshold for "recurrence" (default: 0.1) is arbitrary. Highly sensitive to signal preprocessing and embedding choices. |

---

## 9. Model-Derived Parameters

### 9.1 Softmax Inverse Temperature ($\beta$)

| Field | Value |
|-------|-------|
| **Variable name** | `beta` (from RL model fits) |
| **Classification** | Primary (model-based) |
| **Definition** | Controls the determinism of choice given value differences. |
| **Formula** | $P(A) = \frac{1}{1 + \exp(-\beta (Q_A - Q_B))}$ |
| **Units** | Dimensionless; higher = more deterministic |
| **Interpretation** | $\beta \to 0$: random responding. $\beta \to \infty$: always choose the higher-valued option. Typical values: 1–10. |
| **Threats** | Trades off with learning rate in model fits; poorly identified when value differences are small. Sensitive to the scale of Q-values. |

### 9.2 Learning Rate ($\alpha$)

| Field | Value |
|-------|-------|
| **Variable name** | `alpha` (basic Q-learning), `alpha_pos` / `alpha_neg` (dual-alpha model) |
| **Classification** | Primary (model-based) |
| **Definition** | Step size for value updating: $Q_{t+1}(a) = Q_t(a) + \alpha (r_t - Q_t(a))$ |
| **Units** | Dimensionless, range (0, 1) |
| **Interpretation** | Higher $\alpha$ = faster learning but more noise sensitivity. Lower $\alpha$ = slower, smoother tracking. Asymmetric learning rates ($\alpha_{+} \neq \alpha_{-}$) indicate valence-dependent updating. |
| **Threats** | Highly correlated with $\beta$ in typical datasets. The Q-learning update rule assumes stationary environments; in our task (with phase changes), $\alpha$ reflects an average across contexts. |

### 9.3 Forgetting Rate

| Field | Value |
|-------|-------|
| **Variable name** | `forget` (forgetting Q-learning model) |
| **Classification** | Exploratory (model-based) |
| **Definition** | Rate at which the unchosen option's Q-value decays toward 0.5. |
| **Formula** | $Q_{t+1}(\text{unchosen}) = Q_t(\text{unchosen}) + \phi \cdot (0.5 - Q_t(\text{unchosen}))$ |
| **Units** | Dimensionless, range (0, 1) |
| **Interpretation** | Higher forgetting = faster decay of unchosen option's value, promoting exploration / switching. Captures the intuition that information about an unsampled option becomes stale. |
| **Threats** | Additional free parameter; may overfit in short sessions. Correlated with learning rate. |

### 9.4 Matching Law Sensitivity ($s$) and Bias ($b$)

| Field | Value |
|-------|-------|
| **Variable name** | `sensitivity`, `bias` (matching models) |
| **Classification** | Secondary (model-based) |
| **Definition** | From the generalized matching law: $\log(B_A/B_B) = s \cdot \log(R_A/R_B) + \log(b)$ |
| **Units** | Sensitivity: dimensionless slope. Bias: dimensionless ratio. |
| **Interpretation** | $s = 1$: strict matching. $s < 1$: undermatching (insufficient sensitivity). $s > 1$: overmatching (exaggerated sensitivity). $b \neq 1$: inherent side bias. |
| **Threats** | Requires sufficient variation in reinforcement ratios across windows. Log-ratio is undefined when one option is never chosen or never rewarded in a window — those bins are excluded, potentially biasing estimates. Non-stationarity violates the steady-state assumption of the matching law. |

---

## 10. Model Comparison Metrics

### 10.1 AIC / BIC

| Field | Value |
|-------|-------|
| **Variable name** | `aic`, `bic` |
| **Classification** | Primary (model comparison) |
| **Definition** | Akaike and Bayesian Information Criteria for model selection. |
| **Formula** | $\text{AIC} = 2k + 2 \cdot \text{NLL}$; $\text{BIC} = k \ln(n) + 2 \cdot \text{NLL}$ |
| **Units** | Dimensionless (lower is better) |
| **Interpretation** | AIC favors predictive accuracy; BIC penalizes complexity more heavily. Compare models within participant; the model with lowest BIC/AIC is preferred. |
| **Threats** | AIC/BIC assume the true model is in the candidate set. With few observations per session, BIC's complexity penalty may be too harsh. Neither metric quantifies absolute fit quality. |

### 10.2 Negative Log-Likelihood (NLL)

| Field | Value |
|-------|-------|
| **Variable name** | `nll` |
| **Classification** | Primary (model comparison) |
| **Definition** | Negative log-likelihood of the observed choice sequence under the model. |
| **Formula** | $\text{NLL} = -\sum_t \log P_{\text{model}}(c_t \mid \text{history})$ |
| **Units** | Nats (dimensionless); lower is better |
| **Interpretation** | Measures how surprised the model is by the observed data. Compare to the random-model baseline ($\text{NLL}_{\text{random}} = N \log 2$) and to the bias-model baseline. |
| **Threats** | Sensitive to outlier trials where the model assigns very low probability. Not directly comparable across sessions with different $N$. |

---

## Summary Table

| Variable | Classification | Level | Key Use |
|----------|---------------|-------|---------|
| P(A) | Primary | Session / Event | Choice allocation tracking |
| Reward rate | Primary | Session / Event | Foraging efficiency |
| Proportion optimal | Primary | Session / Event | Hidden-state tracking |
| Switch rate | Primary | Session / Event | Exploration vs exploitation |
| Run length | Primary | Event / Session | Bout structure |
| Win-stay rate | Primary | Session | Reward sensitivity |
| Lose-shift rate | Primary | Session | Punishment sensitivity |
| Adaptation lag | Primary | Session | Speed of behavioral change |
| Choice entropy | Secondary | Session | Allocation variability |
| Regret | Secondary | Event / Session | Opportunity cost |
| ICI | Secondary | Event / Session | Response vigor |
| Phase-specific rates | Secondary | Session | Context-dependent behavior |
| Local reward rates | Secondary | Event | Experienced environment |
| Matching sensitivity | Secondary | Session | Reinforcement sensitivity |
| RL parameters (α, β) | Primary (model) | Session | Learning and decision noise |
| Forgetting rate | Exploratory (model) | Session | Memory decay |
| Choice bias (recent) | Exploratory | Event | Momentary allocation |
| Previous run length | Exploratory | Event | Giving-up patterns |
| Time since reward/switch | Exploratory | Event | Temporal triggers |
| Autocorrelation | Exploratory | Session | Behavioral momentum |
| Hysteresis | Exploratory | Session | Path dependence |
| Recurrence | Exploratory | Session | Dynamical regularity |
| Pulse response | Exploratory | Group | Perturbation sensitivity |
| AIC / BIC / NLL | Primary (comparison) | Session | Model selection |
