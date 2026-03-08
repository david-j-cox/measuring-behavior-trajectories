# Dynamic Foraging Experiment

A browser-based two-option depleting-patch foraging task designed to generate behavioral time-series data revealing dynamic trajectories, transient adaptation, path dependence, oscillation, and switching dynamics.

Participants repeatedly choose between two options whose hidden reward probabilities deplete when harvested and recover when neglected. Hidden regime shifts across four phases prevent stable equilibrium and produce rich temporal dynamics suitable for trajectory-level analysis.

## Architecture

```
+--------------------------------------------------+
|                  PARTICIPANT BROWSER               |
|                                                    |
|  +----------------------------------------------+ |
|  |               index.html                      | |
|  |  Screens: Consent -> Prolific ID ->           | |
|  |  Instructions -> Practice -> Countdown ->     | |
|  |  Main Task (with Leaderboard) -> End          | |
|  +------+------------------+----+----------------+ |
|         |                  |    |                   |
|  +------v------+  +--------v-+  +---v-----------+  |
|  |   ui.js     |  | main.js  |  | style.css     |  |
|  | DOM render, |  | Flow     |  | Layout,       |  |
|  | screens,    |  | control, |  | animations,   |  |
|  | leaderboard,|  | event    |  | responsive    |  |
|  | plot        |  | wiring   |  | design        |  |
|  +------+------+  +----+-----+  +---------------+  |
|         |              |                            |
|    +----v--------------v----+                       |
|    |      state.js          |                       |
|    | Task engine:           |                       |
|    |  - Latent values V_A,  |                       |
|    |    V_B in [0, 1]       |                       |
|    |  - Recovery (time)     |                       |
|    |  - Depletion (click)   |                       |
|    |  - Phase detection     |                       |
|    |  - Bonus pulses        |                       |
|    |  - Reward sampling     |                       |
|    |  - Switch tracking     |                       |
|    +-----+------------------+                       |
|          |                                          |
|    +-----v---------+    +------------------------+  |
|    |  config.js    |    |     logger.js          |  |
|    | All task       |    | Event array,          |  |
|    | parameters,   |    | session metadata,      |  |
|    | phase defs,   |    | JSON/CSV export        |  |
|    | Firebase cfg, |    +----------+-------------+  |
|    | leaderboard   |               |                |
|    +---------------+    +----------v-------------+  |
|                         | firebase-upload.js     |  |
|                         | Firestore write-only   |  |
|                         | upload at task end     |  |
|                         +----------+-------------+  |
+-----------------------------------------------------|
                                     |
                          +----------v-------------+
                          |   Firebase Firestore   |
                          |                        |
                          |   Collection:          |
                          |     sessions/{id}      |
                          |       - metadata {}    |
                          |       - events []      |
                          |                        |
                          |   Rules: create-only   |
                          |   (no read/update/     |
                          |    delete from client)  |
                          +------------------------+
```

### Data Flow

```
Participant clicks Option A or B
        |
        v
    main.js handleClick()
        |
        v
    state.js processClick()
        |-- Recover both options based on elapsed time
        |-- Sample reward from chosen option's latent value
        |-- Deplete chosen option
        |-- Compute switching metrics
        |-- Return full result object
        |
        v
    main.js
        |-- logger.logEvent(result)        --> logger.js events[]
        |-- UI.updateScore()               --> ui.js DOM
        |-- UI.showFeedback()              --> ui.js DOM
        |-- UI.updateLeaderboard()         --> ui.js DOM
        |
        v
    (at task end)
        |-- FirebaseUpload.uploadSession() --> Firestore
        |-- console.log(data)              --> browser console
```

### Experiment Flow

```
[Consent] --> [Prolific ID] --> [Instructions] --> [Practice 20s]
                                                        |
                                                        v
                                                 [Countdown 5s]
                                                        |
                                                        v
                                              [Main Task 6 min]
                                             (with leaderboard)
                                                        |
                                                        v
                                              [End Screen + Upload]
```

## Quick Start

```bash
# From the project root, serve the experiment directory:
python3 -m http.server 8000 --directory experiment

# Then open: http://localhost:8000
```

No build step, no npm, no dependencies. The Firebase SDK is loaded from Google's CDN at runtime.

## File Structure

```
measuring-behavior-trajectories/
  .gitignore
  README.md
  experiment/
    index.html                 Entry point and all screen markup
    css/
      style.css                All styling (screens, buttons, leaderboard, etc.)
    js/
      config.js                All task parameters + Firebase config + leaderboard data
      state.js                 TaskState class: latent values, depletion, recovery, reward
      logger.js                DataLogger class: event logging, JSON/CSV export methods
      firebase-upload.js       FirebaseUpload module: Firestore initialization and upload
      ui.js                    UI module: DOM rendering, screen transitions, leaderboard, plot
      main.js                  Bootstrap IIFE: flow control, click handling, Firebase wiring
```

## How It Works

### Participant Experience

1. **Consent screen** — IRB-style informed consent with a checkbox gate.
2. **Prolific ID entry** — Text input for participant identifier. Also reads `PROLIFIC_PID` from URL query params automatically.
3. **Instructions** — Brief explanation: two buttons, earn points, payoffs may change. No mention of hidden mechanics.
4. **Practice round (20s)** — Identical mechanics to the main task. Score is not recorded. Leaderboard is hidden.
5. **Countdown (5s)** — "Practice Complete" screen with a visible 5-4-3-2-1 countdown.
6. **Main task (6 min)** — Two large buttons (Option A, Option B), a countdown timer, a cumulative score, and a live leaderboard on the right. Clicking produces immediate feedback (+10 points or "No reward").
7. **End screen** — Final score, reward count, click count, a trajectory plot, and a data upload status message.

### Hidden Dynamics

Each option has a **latent value** V in [0, 1] that the participant never sees. This value determines the probability of reward on each click.

**On every click, the following happens in order:**

1. **Recovery** — Both options recover based on elapsed time since the previous click:
   ```
   V_A = min(1, V_A + r_A * dt)
   V_B = min(1, V_B + r_B * dt)
   ```
   where `dt` is seconds since last click and `r` is the recovery rate (per second).

2. **Reward sampling** — The chosen option's current latent value is used as the Bernoulli reward probability. If a Phase 4 bonus pulse is active for the chosen option, its magnitude is added (capped at 1.0):
   ```
   P(reward) = min(1, V_chosen + bonus_if_active)
   reward ~ Bernoulli(P)
   points = reward * 10
   ```

3. **Depletion** — The chosen option is depleted:
   ```
   V_chosen = max(0, V_chosen - d_chosen)
   ```

This creates a **forage-and-deplete** dynamic: harvesting an option reduces future payoff; neglecting it allows recovery. Optimal behavior requires tracking contingencies and switching adaptively.

### Phase Schedule (Hidden from Participant)

| Phase | Time (s) | Description              | r_A  | r_B  | d_A  | d_B  |
|-------|----------|--------------------------|------|------|------|------|
| 1     | 0-90     | Symmetric baseline       | 0.18 | 0.18 | 0.12 | 0.12 |
| 2     | 90-180   | Option A advantage       | 0.24 | 0.10 | 0.12 | 0.12 |
| 3     | 180-270  | Option B advantage       | 0.10 | 0.24 | 0.12 | 0.12 |
| 4     | 270-360  | Scarcity + bonus pulses  | 0.08 | 0.08 | 0.12 | 0.12 |

- **Phase 1** establishes baseline exploration and harvesting patterns.
- **Phases 2 and 3** create asymmetric contingencies that should produce gradual drift toward the richer option, with observable adaptation lag.
- **Phase 4** reduces both recovery rates, creating scarcity. Hidden bonus pulses (3s on, 9s off, alternating A/B) produce intermittent bursts of profitability that destabilize simple habits.

### Bonus Pulse Mechanics (Phase 4 Only)

Every 12 seconds within Phase 4, a 3-second hidden pulse adds +0.20 to the reward probability of one option (alternating A, B, A, B...). The pulse is not announced. It simply makes one option temporarily more profitable:

```
Pulse schedule (relative to Phase 4 start at 270s):
  0-3s:   +0.20 to A
 12-15s:  +0.20 to B
 24-27s:  +0.20 to A
 36-39s:  +0.20 to B
 48-51s:  +0.20 to A
 60-63s:  +0.20 to B
 72-75s:  +0.20 to A
 84-87s:  +0.20 to B
```

### Switching Metrics

Computed on every click:

| Metric | Description |
|--------|-------------|
| `switchFlag` | 1 if the chosen option differs from the previous choice, 0 otherwise |
| `runLengthCurrentOption` | Consecutive same-option choices ending at this click |
| `timeSinceLastSwitchMs` | Milliseconds since the most recent switch event |

### Cooldown

A 200ms cooldown prevents hardware-level double-clicks. During cooldown, buttons are visually dimmed and clicks are ignored.

### Leaderboard

During the main task, a live leaderboard is displayed on the right side of the screen. It shows simulated scores from "previous participants" (configurable in `config.js`) alongside the current participant's live score. The participant's row is highlighted and moves up the ranking as they earn points. This provides social comparison motivation without affecting task mechanics.

## Data Output

### Event-Level Data (one row per click)

Every click during the main task generates a record with the following fields:

| Field | Type | Description |
|-------|------|-------------|
| `participantId` | string | Auto-generated unique ID |
| `sessionId` | string | Auto-generated session ID |
| `timestampMsFromTaskStart` | int | ms since main task began |
| `absoluteTimestamp` | string | ISO 8601 wall-clock time |
| `clickIndex` | int | 1-indexed click number |
| `chosenOption` | string | "A" or "B" |
| `rewardOutcome` | int | 1 = rewarded, 0 = not |
| `pointsEarnedThisClick` | int | 10 or 0 |
| `cumulativeScore` | int | Running total |
| `timeSincePrevClickMs` | int | Inter-click interval in ms |
| `phaseId` | int | 1-4 |
| `phaseLabel` | string | Human-readable phase name |
| `latentValueAPreClick` | float | V_A after recovery, before depletion |
| `latentValueBPreClick` | float | V_B after recovery, before depletion |
| `latentValueAPostClick` | float | V_A after depletion |
| `latentValueBPostClick` | float | V_B after depletion |
| `rewardProbabilityUsed` | float | Actual P(reward) including any bonus |
| `activeBonusPulse` | int | 1 if bonus pulse active for chosen option |
| `bonusTarget` | string | Pulse target ("A", "B", or "") |
| `runLengthCurrentOption` | int | Consecutive same-option count |
| `timeSinceLastSwitchMs` | int | ms since last switch |
| `switchFlag` | int | 1 = switch, 0 = stay |
| `totalClicksSoFar` | int | Cumulative clicks |
| `totalRewardsSoFar` | int | Cumulative rewards |

### Session Metadata

| Field | Description |
|-------|-------------|
| `participantId` | Auto-generated unique ID |
| `prolificId` | Participant-entered Prolific ID |
| `sessionId` | Auto-generated session ID |
| `taskVersion` | Semantic version from config |
| `browser` | User agent string |
| `viewportWidth`, `viewportHeight` | Browser viewport dimensions |
| `startTime`, `endTime` | ISO 8601 timestamps |
| `consentGiven` | Boolean |
| `practiceCompleted` | Boolean |
| `totalDurationCompletedMs` | Main task duration |
| `finalScore` | Total points |
| `totalClicks` | Total click count |
| `totalRewards` | Total reward count |
| `totalSwitches` | Total switch count |
| `overallRewardRate` | Rewards / clicks |

### Data Storage

Data is automatically uploaded to **Firebase Firestore** when the task ends. Each session is stored as a single document in the `sessions` collection:

```
Firestore
  └── sessions/
        └── {sessionId}/
              ├── metadata: { participantId, prolificId, finalScore, ... }
              └── events: [ { clickIndex, chosenOption, rewardOutcome, ... }, ... ]
```

**Firestore security rules** are set to create-only: the browser can write new session documents but cannot read, update, or delete any data.

Data is also logged to the browser console for debugging. The `DataLogger` class retains `downloadJSON()` and `downloadCSV()` methods that can be called from the console if needed.

## Modifying Task Parameters

All parameters live in `experiment/js/config.js`. Nothing is hardcoded elsewhere.

### Core Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `taskVersion` | `"1.0.0"` | Semantic version string |
| `pointsPerReward` | `10` | Points per successful harvest |
| `cooldownMs` | `200` | Minimum ms between clicks |
| `practiceDurationMs` | `20000` | Practice block length (20s) |
| `mainTaskDurationMs` | `360000` | Main task length (6 min) |
| `countdownSec` | `5` | Countdown between practice and main |
| `startingVA` | `0.7` | Initial latent value for Option A |
| `startingVB` | `0.7` | Initial latent value for Option B |

### Phase Definitions

The `phases` array contains objects with:
- `id`, `label` — Phase identifier and human-readable name
- `startMs`, `endMs` — Phase boundaries in ms from main-task start
- `rA`, `rB` — Recovery rates per second for each option
- `dA`, `dB` — Depletion amounts per click for each option

### Bonus Pulse Configuration

The `bonusPulse` object controls Phase 4 pulses:
- `intervalMs` — Time between pulse onsets (default 12000)
- `durationMs` — Pulse duration (default 3000)
- `magnitude` — Added to reward probability (default 0.20)
- `sequence` — Array of targets per pulse (default alternating A/B)

### Leaderboard

The `leaderboard` array contains `{ name, score }` objects representing simulated prior participants. Replace with real data or a backend fetch when available.

### Firebase

`FIREBASE_CONFIG` at the top of `config.js` holds the Firebase project configuration (API key, project ID, etc.). This is client-side config and is designed to be public; security is enforced by Firestore rules, not by hiding the config.

## Firebase Setup

1. Create a Firebase project at [console.firebase.google.com](https://console.firebase.google.com)
2. Register a web app and copy the `firebaseConfig` object into `FIREBASE_CONFIG` in `config.js`
3. Create a Firestore database (production mode)
4. Set security rules:
   ```
   rules_version = '2';
   service cloud.firestore {
     match /databases/{database}/documents {
       match /sessions/{sessionId} {
         allow create: if true;
         allow read, update, delete: if false;
       }
     }
   }
   ```

### Retrieving Data

From the Firebase console:
- **Firestore > sessions** — Browse individual session documents

Programmatically:
- **Firebase Admin SDK** (Python/Node) — Pull all documents for analysis
- **`gcloud firestore export`** — Bulk export to Cloud Storage, then download

## Deployment

This is a static site. The `experiment/` directory can be deployed to any static host:

| Platform | How |
|----------|-----|
| **GitHub Pages** | Push `experiment/` contents to a `gh-pages` branch |
| **Netlify** | Point at the `experiment/` directory |
| **Vercel** | Point at the `experiment/` directory |
| **Firebase Hosting** | `firebase deploy` (can be enabled later in the same project) |
| **S3 + CloudFront** | Upload `experiment/` contents with static hosting |
| **Any server** | `nginx`, `caddy`, or `python3 -m http.server` |

### Prolific Integration

The experiment reads `PROLIFIC_PID` from URL query params automatically. When setting up a Prolific study, use a URL like:

```
https://your-host.com/?PROLIFIC_PID={{%PROLIFIC_PID%}}
```

The ID will be pre-filled on the Prolific ID screen. To add a completion redirect, add logic at the end of `finishExperiment()` in `main.js`.

## Analysis Pipeline

A complete Python analysis pipeline lives in `analysis/`. It pulls data directly from Firestore and runs a 14-step analysis sequence.

### Setup

```bash
cd analysis
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Running

```bash
# Full pipeline
python run_analysis.py --config config.yaml

# Specific steps only
python run_analysis.py --steps validate transform metrics plots phase dynamical fractal individual_differences report

# Skip plots
python run_analysis.py --no-plots
```

### Pipeline Steps

| Step | Name | Description |
|------|------|-------------|
| 1 | Load | Pull events and metadata from Firestore |
| 2 | Validate | Check data integrity, flag/exclude bad sessions |
| 3 | Transform | Compute derived variables (rolling proportions, local rates, etc.) |
| 4 | Metrics | Session-level summary statistics |
| 5 | Plots | Individual session dashboards + group summary figures |
| 6 | Phase analysis | Pre/post transition tests, adaptation lags, hysteresis |
| 7 | Pulse analysis | Perturbation-triggered averages for Phase 4 bonus pulses |
| 8 | Dynamical analysis | State space, autocorrelation, RQA, EDM simplex, CCM, S-Map |
| 9 | Changepoint detection | Bayesian Online Change Point Detection (BOCPD) |
| 10 | Fractal analysis | Detrended Fluctuation Analysis (DFA), sample entropy |
| 11 | Model fitting | Baseline, RL, matching law, and HMM models |
| 12 | Individual differences | PCA + GMM clustering of behavioral phenotypes |
| 13 | Save | Export processed event data |
| 14 | Report | Generate HTML report with all figures and interpretations |

### Output Structure

```
analysis/outputs/
  figures/
    individual/           Per-session dashboards
    group/                Group summary plots
    phase/                Phase transition analysis
    pulse/                Perturbation-triggered averages
    dynamical/            State space, recurrence, EDM, CCM, S-Map
    changepoint/          BOCPD results
    fractal/              DFA and entropy plots
    individual_differences/ PCA biplot, cluster profiles, dendrogram
  tables/
    session_metrics.csv
    model_comparison.csv
    cluster_assignments.csv
    cluster_profiles.csv
    ...
  reports/
    analysis_report.html  Complete analysis report
```

### Key Dependencies

- `pandas`, `numpy`, `scipy` — Data manipulation and statistics
- `matplotlib`, `seaborn` — Plotting
- `scikit-learn` — PCA, GMM clustering
- `pyEDM` — Empirical Dynamical Modeling (simplex projection, CCM, S-Map)
- `firebase-admin` — Firestore data access

## Optional Extensions (Not Yet Implemented)

1. **Switch cost** — Add a brief timeout or point penalty when switching options, creating a travel-cost analog that increases perseveration and makes switching decisions more deliberate.
2. **Hidden context switching** — Occasionally swap which visual button maps to which hidden patch without telling the participant, testing whether they track reward statistics or visual position.
3. **2D cursor foraging** — Replace buttons with spatial patches on a canvas where the participant moves a cursor to harvest, adding travel time as a natural cost.

## Development

```bash
# Local development
python3 -m http.server 8000 --directory experiment

# Branch structure
main    — stable releases
dev     — active development
```

All changes should be developed on `dev` and merged to `main` when validated.
