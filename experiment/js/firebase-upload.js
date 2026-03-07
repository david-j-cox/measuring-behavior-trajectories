// ============================================================
// firebase-upload.js — Firestore data upload
// ============================================================

const FirebaseUpload = {
  db: null,

  init(firebaseConfig) {
    try {
      const app = firebase.initializeApp(firebaseConfig);
      this.db = firebase.firestore(app);
      console.log("Firebase initialized successfully.");
    } catch (err) {
      console.error("Firebase init failed:", err);
    }
  },

  async uploadSession(dataObject) {
    if (!this.db) {
      console.warn("Firebase not initialized — data not uploaded.");
      return { success: false, error: "not_initialized" };
    }

    const sessionId = dataObject.metadata.sessionId || "unknown";

    try {
      await this.db.collection("sessions").doc(sessionId).set(dataObject);
      console.log(`Session ${sessionId} uploaded to Firestore.`);
      return { success: true };
    } catch (err) {
      console.error("Firestore upload failed:", err);
      return { success: false, error: err.message };
    }
  }
};
