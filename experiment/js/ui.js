// ============================================================
// ui.js — All DOM manipulation and rendering
// ============================================================

const UI = {

  // Cache DOM references
  els: {},

  _allScreenIds: [
    "screen-consent", "screen-prolific", "screen-instructions",
    "screen-countdown", "screen-task", "screen-end", "screen-mobile"
  ],

  init() {
    this.els = {
      screenConsent: document.getElementById("screen-consent"),
      screenProlific: document.getElementById("screen-prolific"),
      screenInstructions: document.getElementById("screen-instructions"),
      screenCountdown: document.getElementById("screen-countdown"),
      screenTask: document.getElementById("screen-task"),
      screenEnd: document.getElementById("screen-end"),
      screenMobile: document.getElementById("screen-mobile"),
      consentCheckbox: document.getElementById("consent-checkbox"),
      btnConsent: document.getElementById("btn-consent"),
      prolificInput: document.getElementById("prolific-id-input"),
      btnProlific: document.getElementById("btn-prolific"),
      btnStart: document.getElementById("btn-start"),
      countdownNumber: document.getElementById("countdown-number"),
      btnA: document.getElementById("btn-a"),
      btnB: document.getElementById("btn-b"),
      timer: document.getElementById("timer"),
      score: document.getElementById("score"),
      feedback: document.getElementById("feedback"),
      phaseLabel: document.getElementById("phase-label"),
      leaderboardList: document.getElementById("leaderboard-list"),
      endScore: document.getElementById("end-score"),
      endRewards: document.getElementById("end-rewards"),
      endClicks: document.getElementById("end-clicks"),
      completionCodeBox: document.getElementById("completion-code-box"),
      completionCode: document.getElementById("completion-code")
    };
  },

  // ---- Screen transitions ----

  showScreen(name) {
    for (const id of this._allScreenIds) {
      document.getElementById(id).classList.add("hidden");
    }
    const map = {
      consent: this.els.screenConsent,
      prolific: this.els.screenProlific,
      instructions: this.els.screenInstructions,
      countdown: this.els.screenCountdown,
      task: this.els.screenTask,
      end: this.els.screenEnd,
      mobile: this.els.screenMobile
    };
    if (map[name]) map[name].classList.remove("hidden");
  },

  // ---- Task screen updates ----

  updateTimer(remainingMs) {
    const totalSec = Math.max(0, Math.ceil(remainingMs / 1000));
    const min = Math.floor(totalSec / 60);
    const sec = totalSec % 60;
    this.els.timer.textContent = `${min}:${sec.toString().padStart(2, "0")}`;
  },

  updateScore(points) {
    this.els.score.textContent = points;
  },

  showFeedback(rewarded, points) {
    const el = this.els.feedback;
    if (rewarded) {
      el.textContent = `+${points} points`;
      el.className = "feedback reward";
    } else {
      el.textContent = "No reward";
      el.className = "feedback miss";
    }
    // Clear after brief display
    clearTimeout(this._feedbackTimeout);
    this._feedbackTimeout = setTimeout(() => {
      el.textContent = "\u00A0"; // non-breaking space to hold layout
      el.className = "feedback";
    }, 600);
  },

  flashButton(btnEl, rewarded) {
    const cls = rewarded ? "flash-reward" : "flash-miss";
    btnEl.classList.add(cls);
    setTimeout(() => btnEl.classList.remove(cls), 250);
  },

  setPhaseLabel(text) {
    if (this.els.phaseLabel) {
      this.els.phaseLabel.textContent = text;
    }
  },

  setCooldown(active) {
    if (active) {
      this.els.btnA.classList.add("cooldown");
      this.els.btnB.classList.add("cooldown");
    } else {
      this.els.btnA.classList.remove("cooldown");
      this.els.btnB.classList.remove("cooldown");
    }
  },

  // ---- End screen ----

  showEndStats(score, rewards, clicks) {
    this.els.endScore.textContent = score;
    this.els.endRewards.textContent = rewards;
    this.els.endClicks.textContent = clicks;
  },

  // ---- Countdown ----

  updateCountdown(sec) {
    this.els.countdownNumber.textContent = sec;
  },

  // ---- Leaderboard ----

  initLeaderboard(entries, playerLabel) {
    // entries: [{name, score}], playerLabel: string
    this._lbEntries = entries.map(e => ({ ...e }));
    this._lbPlayerLabel = playerLabel;
    this._lbPlayerScore = 0;
    this.updateLeaderboard(0);
  },

  updateLeaderboard(playerScore) {
    this._lbPlayerScore = playerScore;
    // Build combined list
    const all = [
      ...this._lbEntries,
      { name: this._lbPlayerLabel, score: playerScore, isPlayer: true }
    ];
    all.sort((a, b) => b.score - a.score);

    const list = this.els.leaderboardList;
    list.innerHTML = "";
    for (let i = 0; i < all.length; i++) {
      const row = document.createElement("div");
      row.className = "lb-row" + (all[i].isPlayer ? " lb-you" : "");
      row.innerHTML =
        `<span class="lb-rank">${i + 1}.</span>` +
        `<span class="lb-name">${all[i].isPlayer ? "You" : all[i].name}</span>` +
        `<span class="lb-score">${all[i].score.toLocaleString()}</span>`;
      list.appendChild(row);
    }
  },

  // ---- Completion code ----

  showCompletionCode(code) {
    if (code && this.els.completionCodeBox) {
      this.els.completionCode.textContent = code;
      this.els.completionCodeBox.style.display = "block";
    }
  }
};
