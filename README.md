# Dynamic Foraging Experiment

A browser-based two-option depleting-patch foraging task designed to generate behavioral time-series data revealing dynamic trajectories, transient adaptation, path dependence, oscillation, and switching dynamics.

## Quick Start

```bash
# From the project root, serve the experiment directory:
python3 -m http.server 8000 --directory experiment

# Then open: http://localhost:8000
```

No build step, no dependencies. Just a static file server.

## File Structure

```
experiment/
  index.html          Main entry point
  css/style.css       All styling
  js/
    config.js         All task parameters (durations, rates, phases)
    state.js          Task engine: latent values, depletion, recovery, reward
    logger.js         Event logging and JSON/CSV export
    ui.js             DOM rendering and screen management
    main.js           Bootstrap, flow control, click handling
```

## How It Works

### Participant Experience

The participant sees two buttons (Option A, Option B), a countdown timer, and a cumulative score. Clicking either button may earn +10 points or nothing. Instructions say only that payoff patterns may change over time. A 20-second practice round is followed by a 6-minute main task.

### Hidden Dynamics

Each option has a **latent value** V in [0, 1] that the participant never sees. This value determines the probability of reward on each click.

**Recovery:** Both options continuously recover toward 1.0 when left alone. On every click, both options first recover based on elapsed time since the previous click:

```
V_option = min(1, V_option + r_option * dt)
```

where `dt` is seconds since last click and `r_option` is the recovery rate (per second).

**Reward sampling:** After recovery, the chosen option's current latent value is used as the Bernoulli reward probability. If a Phase 4 bonus pulse is active for the chosen option, its magnitude is added (capped at 1.0).

```
P(reward) = min(1, V_chosen + bonus_if_active)
reward ~ Bernoulli(P)
```

**Depletion:** After reward sampling, the chosen option is depleted:

```
V_chosen = max(0, V_chosen - d_chosen)
```

This creates a **forage-and-deplete** dynamic: harvesting an option reduces future payoff; neglecting it allows recovery.

### Phase Schedule (Hidden from Participant)

| Phase | Time (s)  | Description | rA   | rB   | dA   | dB   |
|-------|-----------|-------------|------|------|------|------|
| 1     | 0–90      | Symmetric baseline | 0.18 | 0.18 | 0.12 | 0.12 |
| 2     | 90–180    | A advantage  | 0.24 | 0.10 | 0.12 | 0.12 |
| 3     | 180–270   | B advantage  | 0.10 | 0.24 | 0.12 | 0.12 |
| 4     | 270–360   | Scarcity + bonus pulses | 0.08 | 0.08 | 0.12 | 0.12 |

**Phase 4 bonus pulses:** Every 12 seconds, a 3-second hidden pulse adds +0.20 to one option's reward probability (alternating A, B, A, B...). These are not announced to the participant.

### Switching Metrics

On every click:
- **Switch flag:** 1 if the chosen option differs from the previous choice, 0 otherwise.
- **Run length:** consecutive same-option choices ending at this click.
- **Time since last switch:** milliseconds since the most recent switch event.

### Cooldown

A 200ms cooldown prevents hardware-level double-clicks. During cooldown, buttons are visually dimmed and clicks are ignored.

## Modifying Task Parameters

All parameters live in `experiment/js/config.js`:

- `mainTaskDurationMs` — total task length (default 360000 = 6 min)
- `practiceDurationMs` — practice block length (default 20000 = 20 s)
- `pointsPerReward` — points per successful harvest (default 10)
- `cooldownMs` — inter-click cooldown (default 200)
- `startingVA`, `startingVB` — initial latent values (default 0.7)
- `phases[]` — array of phase objects with start/end times, recovery rates, depletion amounts
- `bonusPulse` — Phase 4 pulse timing, duration, magnitude, and target sequence
- `practice` — recovery/depletion rates during the practice block

## Data Output

### Event-level data (one row per click)

Every click in the main task generates a record with:

| Field | Description |
|-------|-------------|
| `participantId` | Auto-generated unique ID |
| `sessionId` | Auto-generated session ID |
| `timestampMsFromTaskStart` | ms since main task began |
| `absoluteTimestamp` | ISO 8601 wall-clock time |
| `clickIndex` | 1-indexed click number |
| `chosenOption` | "A" or "B" |
| `rewardOutcome` | 1 = rewarded, 0 = not |
| `pointsEarnedThisClick` | 10 or 0 |
| `cumulativeScore` | running total |
| `timeSincePrevClickMs` | ICI in ms |
| `phaseId` | 1–4 |
| `phaseLabel` | human-readable phase name |
| `latentValueAPreClick` | V_A before depletion |
| `latentValueBPreClick` | V_B before depletion |
| `latentValueAPostClick` | V_A after depletion |
| `latentValueBPostClick` | V_B after depletion |
| `rewardProbabilityUsed` | actual P(reward) including bonus |
| `activeBonusPulse` | 1 if pulse active for chosen, else 0 |
| `bonusTarget` | pulse target option or "" |
| `runLengthCurrentOption` | consecutive same-option count |
| `timeSinceLastSwitchMs` | ms since last switch |
| `switchFlag` | 1 = switch, 0 = stay |
| `totalClicksSoFar` | cumulative clicks |
| `totalRewardsSoFar` | cumulative rewards |

### Session metadata

Includes participant/session IDs, browser info, viewport size, task version, timestamps, practice completion, final score, total clicks, switches, and overall reward rate.

### Export

At task end, two download buttons appear:
- **Download JSON** — full data object (metadata + events array)
- **Download CSV** — flat event-level table

Data is also logged to the browser console via `console.log()`.

## Deployment

This is a static site. Deploy to any static host:

- **GitHub Pages:** push `experiment/` to a `gh-pages` branch or configure Pages to serve from a subdirectory
- **Netlify / Vercel:** point at the `experiment/` directory
- **S3 + CloudFront:** upload `experiment/` contents to a bucket with static hosting enabled
- **Any web server:** `nginx`, `caddy`, or `python3 -m http.server`

For adding server-side data collection later, the `DataLogger` class in `logger.js` can be extended with a `uploadToServer(url)` method that POSTs the full data object. The core task engine (`state.js`) needs no changes.

## Extending for Production

The architecture is designed so that you can add these without rewriting the task engine:

- **Participant ID entry / consent screen:** Add a new screen div in `index.html`, gate the start button behind consent + ID entry, pass the ID to `logger.setSessionMeta()`
- **Backend upload:** Add a `fetch()` POST in `main.js` at experiment end
- **Prolific / MTurk integration:** Read participant ID from URL params, redirect to completion URL at end

## Intended Analyses

The data structure supports:
- Choice allocation trajectories over time
- Adaptation lag after hidden contingency shifts
- Switching dynamics and run-length distributions
- Path dependence and hysteresis across phases
- Trial-by-trial model fitting (RL, melioration, matching law, nonlinear dynamical models)
- Inter-click interval analysis

## Optional Extensions (Not Yet Implemented)

1. **Switch cost:** Add a brief timeout or point penalty when switching options, creating a travel-cost analog that should increase perseveration and make switching decisions more deliberate.
2. **Hidden context switching:** Occasionally swap which visual button maps to which hidden patch without telling the participant, testing whether they track reward statistics or visual position.
3. **2D cursor foraging:** Replace buttons with spatial patches on a canvas where the participant moves a cursor to harvest, adding travel time as a natural cost.
