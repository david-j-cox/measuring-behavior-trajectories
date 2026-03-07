// ============================================================
// config.js — All task parameters in one place
// ============================================================

const CONFIG = Object.freeze({

  // --- General ---
  taskVersion: "1.0.0",
  pointsPerReward: 10,
  cooldownMs: 200,          // minimum ms between clicks
  practiceDurationMs: 20000, // 20-second practice block
  mainTaskDurationMs: 360000, // 6 minutes

  // --- Starting latent values ---
  startingVA: 0.7,
  startingVB: 0.7,

  // --- Phase schedule (times in ms from main-task start) ---
  phases: [
    {
      id: 1,
      label: "Symmetric baseline",
      startMs: 0,
      endMs: 90000,
      rA: 0.18, rB: 0.18,
      dA: 0.12, dB: 0.12
    },
    {
      id: 2,
      label: "Option A advantage",
      startMs: 90000,
      endMs: 180000,
      rA: 0.24, rB: 0.10,
      dA: 0.12, dB: 0.12
    },
    {
      id: 3,
      label: "Option B advantage",
      startMs: 180000,
      endMs: 270000,
      rA: 0.10, rB: 0.24,
      dA: 0.12, dB: 0.12
    },
    {
      id: 4,
      label: "Scarcity + bonus pulses",
      startMs: 270000,
      endMs: 360000,
      rA: 0.08, rB: 0.08,
      dA: 0.12, dB: 0.12
    }
  ],

  // --- Phase 4 bonus pulses ---
  bonusPulse: {
    intervalMs: 12000,       // one pulse every 12 s
    durationMs: 3000,        // each pulse lasts 3 s
    magnitude: 0.20,         // added to reward probability
    // Predetermined sequence of targets within Phase 4.
    // Pulses alternate A, B, A, B, ...
    sequence: ["A", "B", "A", "B", "A", "B", "A", "B"]
  },

  // --- Inter-condition countdown ---
  countdownSec: 5,

  // --- Prolific completion code ---
  // Shown to participants on the final screen after data upload.
  // Generate this in your Prolific study setup.
  prolificCompletionCode: "CXXXXXXX",

  // --- Minimum viewport width to allow participation ---
  minViewportWidth: 800,

  // --- Practice block uses Phase-1 parameters ---
  practice: {
    rA: 0.18, rB: 0.18,
    dA: 0.12, dB: 0.12
  },

  // --- Leaderboard (simulated past participants) ---
  // These are static simulated scores shown during the task.
  // They do NOT update from real participant data.
  // Replace with a backend fetch when available.
  leaderboard: [
    { name: "Player 3",  score: 5327 },
    { name: "Player 12", score: 4890 },
    { name: "Player 7",  score: 4510 },
    { name: "Player 15", score: 4180 },
    { name: "Player 9",  score: 3820 },
    { name: "Player 1",  score: 3470 },
    { name: "Player 11", score: 3150 },
    { name: "Player 5",  score: 2880 },
    { name: "Player 8",  score: 2590 },
    { name: "Player 14", score: 2340 },
    { name: "Player 6",  score: 2150 },
    { name: "Player 2",  score: 2000 }
  ]
});
