// ============================================================
// logger.js — Data logging and export (JSON + CSV)
// ============================================================

class DataLogger {
  constructor() {
    this.events = [];
    this.sessionMeta = {};
  }

  reset() {
    this.events = [];
    this.sessionMeta = {};
  }

  setSessionMeta(meta) {
    this.sessionMeta = { ...this.sessionMeta, ...meta };
  }

  logEvent(record) {
    const enriched = {
      participantId: this.sessionMeta.participantId || "unknown",
      sessionId: this.sessionMeta.sessionId || "unknown",
      timestampMsFromTaskStart: record.timestampMsFromTaskStart,
      absoluteTimestamp: new Date().toISOString(),
      ...record
    };
    this.events.push(enriched);
  }

  getFullDataObject() {
    return {
      metadata: this.sessionMeta,
      events: this.events
    };
  }

  // ---- JSON export ----

  toJSON() {
    return JSON.stringify(this.getFullDataObject(), null, 2);
  }

  downloadJSON(filename) {
    const blob = new Blob([this.toJSON()], { type: "application/json" });
    this._downloadBlob(blob, filename || "experiment_data.json");
  }

  // ---- CSV export ----

  toCSV() {
    if (this.events.length === 0) return "";
    const headers = Object.keys(this.events[0]);
    const rows = this.events.map(e =>
      headers.map(h => {
        const val = e[h];
        if (val === null || val === undefined) return "";
        if (typeof val === "string" && val.includes(",")) return `"${val}"`;
        return val;
      }).join(",")
    );
    return [headers.join(","), ...rows].join("\n");
  }

  downloadCSV(filename) {
    const blob = new Blob([this.toCSV()], { type: "text/csv" });
    this._downloadBlob(blob, filename || "experiment_data.csv");
  }

  // ---- Helper ----

  _downloadBlob(blob, filename) {
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  }
}
