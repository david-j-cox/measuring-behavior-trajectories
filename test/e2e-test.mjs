// ============================================================
// End-to-end test: simulate full experiment + upload to Firestore
// Run: node test/e2e-test.mjs
// ============================================================

import { initializeApp } from "firebase/app";
import { getFirestore, doc, setDoc } from "firebase/firestore";
import { readFileSync } from "fs";
import vm from "vm";

// ---- Load config and state engine ----
// Use Function constructor to eval browser globals into local scope
function evalBrowserFile(filepath) {
  return readFileSync(filepath, "utf8");
}

const combined = `
  ${evalBrowserFile("experiment/js/firebase-config.js")}
  ${evalBrowserFile("experiment/js/config.js")}
  ${evalBrowserFile("experiment/js/state.js")}
  return { FIREBASE_CONFIG, CONFIG, TaskState };
`;
const { FIREBASE_CONFIG, CONFIG, TaskState } = new Function(combined)();

// ---- Initialize Firebase ----
console.log("1. Initializing Firebase...");
const app = initializeApp(FIREBASE_CONFIG);
const db = getFirestore(app);
console.log("   Firebase initialized.");

// ---- Simulate full experiment ----
console.log("\n2. Simulating full 6-minute experiment...");
const state = new TaskState(CONFIG);
state.isPractice = false;

const sessionId = "test_" + Date.now().toString(36);
const events = [];
let elapsedMs = 0;
const totalMs = CONFIG.mainTaskDurationMs;

let prevChoice = null;
let clickCount = 0;

while (elapsedMs < totalMs) {
  elapsedMs += 300 + Math.random() * 300;
  if (elapsedMs >= totalMs) break;

  let choice;
  if (prevChoice === null) {
    choice = Math.random() < 0.5 ? "A" : "B";
  } else {
    choice = Math.random() < 0.7 ? (prevChoice === "A" ? "B" : "A") : prevChoice;
  }

  const result = state.processClick(choice, elapsedMs);
  clickCount++;

  events.push({
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

  prevChoice = choice;
}

console.log(`   Simulated ${clickCount} clicks across ${(totalMs / 1000)}s`);
console.log(`   Final score: ${state.cumulativeScore}`);
console.log(`   Total rewards: ${state.totalRewards}`);
console.log(`   Total switches: ${state.totalSwitches}`);
console.log(`   Reward rate: ${(state.totalRewards / clickCount * 100).toFixed(1)}%`);

// Phase breakdown
const phaseCounts = {};
for (const e of events) {
  const key = `Phase ${e.phaseId}`;
  if (!phaseCounts[key]) phaseCounts[key] = { clicks: 0, rewards: 0, aChoices: 0 };
  phaseCounts[key].clicks++;
  phaseCounts[key].rewards += e.rewardOutcome;
  if (e.chosenOption === "A") phaseCounts[key].aChoices++;
}
console.log("\n   Phase breakdown:");
for (const [phase, data] of Object.entries(phaseCounts)) {
  const propA = (data.aChoices / data.clicks * 100).toFixed(1);
  const rwdRate = (data.rewards / data.clicks * 100).toFixed(1);
  console.log(`   ${phase}: ${data.clicks} clicks, ${rwdRate}% reward rate, ${propA}% chose A`);
}

// ---- Build data object ----
const dataObject = {
  metadata: {
    participantId: "test_participant",
    prolificId: "TEST_E2E",
    sessionId: sessionId,
    taskVersion: CONFIG.taskVersion,
    browser: "Node.js e2e test",
    viewportWidth: 1920,
    viewportHeight: 1080,
    startTime: new Date().toISOString(),
    endTime: new Date().toISOString(),
    consentGiven: true,
    practiceCompleted: true,
    totalDurationCompletedMs: totalMs,
    finalScore: state.cumulativeScore,
    totalClicks: clickCount,
    totalRewards: state.totalRewards,
    totalSwitches: state.totalSwitches,
    overallRewardRate: +(state.totalRewards / clickCount).toFixed(4),
    isTestSession: true
  },
  events: events
};

// ---- Upload to Firestore ----
console.log(`\n3. Uploading session "${sessionId}" to Firestore...`);
console.log(`   Document size: ${JSON.stringify(dataObject).length} bytes, ${events.length} events`);
try {
  await setDoc(doc(db, "sessions", sessionId), dataObject);
  console.log("   Upload successful!");
  console.log(`\n   Check Firebase console: Firestore > sessions > ${sessionId}`);
  console.log("\n   ALL TESTS PASSED");
} catch (err) {
  console.error("   Upload FAILED:", err.message);
  console.error("\n   TEST FAILED -- check Firebase config and Firestore rules");
  process.exit(1);
}

process.exit(0);
