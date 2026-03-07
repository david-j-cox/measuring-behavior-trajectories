// ============================================================
// state.js — Task engine: latent values, phases, reward logic
// ============================================================

class TaskState {
  constructor(config) {
    this.config = config;
    this.reset();
  }

  reset() {
    this.vA = this.config.startingVA;
    this.vB = this.config.startingVB;
    this.lastClickTimeMs = 0;      // ms from block start
    this.clickIndex = 0;
    this.totalRewards = 0;
    this.cumulativeScore = 0;
    this.previousChoice = null;
    this.runLength = 0;
    this.lastSwitchTimeMs = 0;
    this.totalSwitches = 0;
    this.isPractice = false;
  }

  // ---- Phase helpers ----

  getPhase(elapsedMs) {
    if (this.isPractice) return null;
    for (const p of this.config.phases) {
      if (elapsedMs >= p.startMs && elapsedMs < p.endMs) return p;
    }
    // After last phase, return last phase
    return this.config.phases[this.config.phases.length - 1];
  }

  getParams(elapsedMs) {
    if (this.isPractice) {
      return this.config.practice;
    }
    const phase = this.getPhase(elapsedMs);
    return { rA: phase.rA, rB: phase.rB, dA: phase.dA, dB: phase.dB };
  }

  // ---- Bonus pulse logic (Phase 4 only) ----

  getBonusPulseInfo(elapsedMs) {
    const bp = this.config.bonusPulse;
    const phase = this.getPhase(elapsedMs);
    if (!phase || phase.id !== 4) {
      return { active: false, target: null, magnitude: 0 };
    }
    const phaseElapsed = elapsedMs - phase.startMs;
    const pulseIndex = Math.floor(phaseElapsed / bp.intervalMs);
    const withinPulse = phaseElapsed - (pulseIndex * bp.intervalMs);

    if (withinPulse < bp.durationMs && pulseIndex < bp.sequence.length) {
      return {
        active: true,
        target: bp.sequence[pulseIndex],
        magnitude: bp.magnitude
      };
    }
    return { active: false, target: null, magnitude: 0 };
  }

  // ---- Core click processing ----

  processClick(chosenOption, elapsedMs) {
    const dtSec = (elapsedMs - this.lastClickTimeMs) / 1000;
    const params = this.getParams(elapsedMs);
    const phase = this.getPhase(elapsedMs);

    // --- Recovery: unchosen option recovers over dt ---
    // Both options recover passively; the chosen one then gets depleted.
    this.vA = Math.min(1, this.vA + params.rA * dtSec);
    this.vB = Math.min(1, this.vB + params.rB * dtSec);

    const preClickVA = this.vA;
    const preClickVB = this.vB;

    // --- Reward probability = latent value BEFORE depletion ---
    let rewardProb = chosenOption === "A" ? this.vA : this.vB;

    // Apply bonus pulse if active
    const pulse = this.getBonusPulseInfo(elapsedMs);
    let effectivePulse = { active: false, target: null };
    if (pulse.active && pulse.target === chosenOption) {
      rewardProb = Math.min(1, rewardProb + pulse.magnitude);
      effectivePulse = pulse;
    }

    // --- Bernoulli reward sample ---
    const rewarded = Math.random() < rewardProb;
    const pointsEarned = rewarded ? this.config.pointsPerReward : 0;

    // --- Depletion of chosen option ---
    if (chosenOption === "A") {
      this.vA = Math.max(0, this.vA - params.dA);
    } else {
      this.vB = Math.max(0, this.vB - params.dB);
    }

    const postClickVA = this.vA;
    const postClickVB = this.vB;

    // --- Switching metrics ---
    const isSwitch = this.previousChoice !== null && this.previousChoice !== chosenOption;
    if (isSwitch) {
      this.runLength = 1;
      this.lastSwitchTimeMs = elapsedMs;
      this.totalSwitches++;
    } else {
      this.runLength = this.previousChoice === null ? 1 : this.runLength + 1;
    }

    const timeSinceSwitch = this.previousChoice === null
      ? 0
      : elapsedMs - this.lastSwitchTimeMs;
    const timeSincePrevClick = this.clickIndex === 0
      ? 0
      : elapsedMs - this.lastClickTimeMs;

    this.clickIndex++;
    this.cumulativeScore += pointsEarned;
    if (rewarded) this.totalRewards++;
    this.lastClickTimeMs = elapsedMs;
    const prevChoice = this.previousChoice;
    this.previousChoice = chosenOption;

    return {
      clickIndex: this.clickIndex,
      chosenOption,
      rewardProbabilityUsed: rewardProb,
      rewarded,
      pointsEarned,
      cumulativeScore: this.cumulativeScore,
      timeSincePrevClickMs: timeSincePrevClick,
      phaseId: phase ? phase.id : 0,
      phaseLabel: phase ? phase.label : "practice",
      preClickVA, preClickVB,
      postClickVA, postClickVB,
      activeBonusPulse: effectivePulse.active,
      bonusTarget: effectivePulse.active ? effectivePulse.target : null,
      runLength: this.runLength,
      timeSinceSwitchMs: timeSinceSwitch,
      switchFlag: isSwitch ? 1 : 0,
      totalClicks: this.clickIndex,
      totalRewards: this.totalRewards,
      totalSwitches: this.totalSwitches
    };
  }
}
