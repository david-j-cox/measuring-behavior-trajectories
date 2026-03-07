// ============================================================
// main.js — Bootstrap and task flow controller
// ============================================================

(function () {
  "use strict";

  // ---- Instances ----
  const state = new TaskState(CONFIG);
  const logger = new DataLogger();
  let timerInterval = null;
  let blockStartTime = null;   // performance.now() at block start
  let blockDurationMs = 0;
  let inCooldown = false;
  let taskRunning = false;
  let currentMode = null; // "practice" | "main"
  let prolificId = "";

  // ---- Generate IDs ----

  function generateId() {
    return Date.now().toString(36) + Math.random().toString(36).slice(2, 8);
  }

  // ---- Initialization ----

  function init() {
    UI.init();
    UI.showScreen("consent");

    // Try to read Prolific ID from URL params
    const urlParams = new URLSearchParams(window.location.search);
    const urlPid = urlParams.get("PROLIFIC_PID") || urlParams.get("prolific_pid") || "";

    // Consent screen
    UI.els.consentCheckbox.addEventListener("change", () => {
      UI.els.btnConsent.disabled = !UI.els.consentCheckbox.checked;
    });
    UI.els.btnConsent.addEventListener("click", () => {
      UI.showScreen("prolific");
      // Pre-fill if from URL
      if (urlPid) {
        UI.els.prolificInput.value = urlPid;
        UI.els.btnProlific.disabled = false;
      }
      UI.els.prolificInput.focus();
    });

    // Prolific ID screen
    UI.els.prolificInput.addEventListener("input", () => {
      UI.els.btnProlific.disabled = UI.els.prolificInput.value.trim().length === 0;
    });
    UI.els.btnProlific.addEventListener("click", () => {
      prolificId = UI.els.prolificInput.value.trim();
      UI.showScreen("instructions");
    });
    // Allow Enter key on prolific input
    UI.els.prolificInput.addEventListener("keydown", (e) => {
      if (e.key === "Enter" && UI.els.prolificInput.value.trim().length > 0) {
        UI.els.btnProlific.click();
      }
    });

    // Instructions → practice
    UI.els.btnStart.addEventListener("click", startPractice);

    // Choice buttons
    UI.els.btnA.addEventListener("click", () => handleClick("A"));
    UI.els.btnB.addEventListener("click", () => handleClick("B"));

    // End screen
    UI.els.btnRestart.addEventListener("click", restart);

    // Initialize Firebase
    FirebaseUpload.init(FIREBASE_CONFIG);
  }

  // ---- Practice block ----

  function startPractice() {
    const sid = generateId();
    logger.reset();
    logger.setSessionMeta({
      participantId: generateId(),
      prolificId: prolificId,
      sessionId: sid,
      taskVersion: CONFIG.taskVersion,
      browser: navigator.userAgent,
      viewportWidth: window.innerWidth,
      viewportHeight: window.innerHeight,
      startTime: new Date().toISOString(),
      consentGiven: true,
      practiceCompleted: false
    });

    state.reset();
    state.isPractice = true;
    currentMode = "practice";
    blockDurationMs = CONFIG.practiceDurationMs;

    UI.showScreen("task");
    UI.setPhaseLabel("Practice");
    UI.updateScore(0);
    // Hide leaderboard during practice
    document.getElementById("leaderboard").classList.add("hidden");

    startBlock();
  }

  // ---- Countdown between practice and main ----

  function startCountdown() {
    UI.showScreen("countdown");
    let remaining = CONFIG.countdownSec;
    UI.updateCountdown(remaining);

    const countdownInterval = setInterval(() => {
      remaining--;
      if (remaining <= 0) {
        clearInterval(countdownInterval);
        startMainTask();
      } else {
        UI.updateCountdown(remaining);
      }
    }, 1000);
  }

  // ---- Main task ----

  function startMainTask() {
    state.reset();
    state.isPractice = false;
    currentMode = "main";
    blockDurationMs = CONFIG.mainTaskDurationMs;

    logger.setSessionMeta({ practiceCompleted: true });
    UI.showScreen("task");
    UI.setPhaseLabel("");
    UI.updateScore(0);
    UI.els.feedback.textContent = "\u00A0";

    // Show and initialize leaderboard
    document.getElementById("leaderboard").classList.remove("hidden");
    UI.initLeaderboard(CONFIG.leaderboard, "You");

    startBlock();
  }

  // ---- Block control ----

  function startBlock() {
    blockStartTime = performance.now();
    taskRunning = true;

    // Start countdown timer
    updateTimerDisplay();
    timerInterval = setInterval(updateTimerDisplay, 250);
  }

  function updateTimerDisplay() {
    const elapsed = performance.now() - blockStartTime;
    const remaining = blockDurationMs - elapsed;

    if (remaining <= 0) {
      endBlock();
      return;
    }
    UI.updateTimer(remaining);
  }

  function endBlock() {
    taskRunning = false;
    clearInterval(timerInterval);
    UI.updateTimer(0);

    if (currentMode === "practice") {
      // Transition to countdown, then main task
      startCountdown();
    } else {
      finishExperiment();
    }
  }

  // ---- Click handler ----

  function handleClick(option) {
    if (!taskRunning || inCooldown) return;

    const now = performance.now();
    const elapsedMs = now - blockStartTime;

    // Process through state engine
    const result = state.processClick(option, elapsedMs);

    // Log event (main task only)
    if (currentMode === "main") {
      logger.logEvent({
        timestampMsFromTaskStart: Math.round(elapsedMs),
        clickIndex: result.clickIndex,
        chosenOption: result.chosenOption,
        rewardOutcome: result.rewarded ? 1 : 0,
        pointsEarnedThisClick: result.pointsEarned,
        cumulativeScore: result.cumulativeScore,
        timeSincePrevClickMs: Math.round(result.timeSincePrevClickMs),
        phaseId: result.phaseId,
        phaseLabel: result.phaseLabel,
        latentValueAPreClick: +result.preClickVA.toFixed(4),
        latentValueBPreClick: +result.preClickVB.toFixed(4),
        latentValueAPostClick: +result.postClickVA.toFixed(4),
        latentValueBPostClick: +result.postClickVB.toFixed(4),
        rewardProbabilityUsed: +result.rewardProbabilityUsed.toFixed(4),
        activeBonusPulse: result.activeBonusPulse ? 1 : 0,
        bonusTarget: result.bonusTarget || "",
        runLengthCurrentOption: result.runLength,
        timeSinceLastSwitchMs: Math.round(result.timeSinceSwitchMs),
        switchFlag: result.switchFlag,
        totalClicksSoFar: result.totalClicks,
        totalRewardsSoFar: result.totalRewards
      });

      // Update leaderboard
      UI.updateLeaderboard(result.cumulativeScore);
    }

    // Update UI
    UI.updateScore(result.cumulativeScore);
    UI.showFeedback(result.rewarded, result.pointsEarned);
    const btnEl = option === "A" ? UI.els.btnA : UI.els.btnB;
    UI.flashButton(btnEl, result.rewarded);

    // Cooldown
    inCooldown = true;
    UI.setCooldown(true);
    setTimeout(() => {
      inCooldown = false;
      UI.setCooldown(false);
    }, CONFIG.cooldownMs);
  }

  // ---- Experiment end ----

  function finishExperiment() {
    const finalScore = state.cumulativeScore;
    const finalRewards = state.totalRewards;
    const finalClicks = state.clickIndex;

    logger.setSessionMeta({
      endTime: new Date().toISOString(),
      totalDurationCompletedMs: CONFIG.mainTaskDurationMs,
      finalScore,
      totalClicks: finalClicks,
      totalRewards: finalRewards,
      totalSwitches: state.totalSwitches,
      overallRewardRate: finalClicks > 0 ? +(finalRewards / finalClicks).toFixed(4) : 0
    });

    UI.showScreen("end");
    UI.showEndStats(finalScore, finalRewards, finalClicks);
    UI.drawPlot(logger.events, CONFIG);

    // Dump to console for debugging
    const dataObject = logger.getFullDataObject();
    console.log("=== Experiment Data ===");
    console.log(dataObject);

    // Upload to Firestore
    const statusEl = document.getElementById("upload-status");
    FirebaseUpload.uploadSession(dataObject).then(result => {
      if (result.success) {
        statusEl.textContent = "Data saved successfully.";
        statusEl.classList.add("upload-success");
      } else {
        statusEl.textContent = "Data could not be saved. Please contact the researcher.";
        statusEl.classList.add("upload-error");
      }
    });
  }

  // ---- Restart ----

  function restart() {
    clearInterval(timerInterval);
    taskRunning = false;
    inCooldown = false;
    currentMode = null;
    UI.showScreen("consent");
  }

  // ---- Boot ----
  document.addEventListener("DOMContentLoaded", init);
})();
