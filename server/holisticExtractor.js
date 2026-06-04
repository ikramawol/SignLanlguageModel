// holisticExtractor.js
// Extracts a 233-dim landmark vector from MediaPipe Holistic JS results.
// Layout matches extract_keypoints.py and medibridge_record.py exactly:
//   [  0: 99]  pose landmarks        33 × 3 (x, y, z)
//   [ 99:162]  left-hand normalised  21 × 3
//   [162:225]  right-hand normalised 21 × 3
//   [225:229]  left-wrist  body anchor  4 values
//   [229:233]  right-wrist body anchor  4 values

const POSE_COUNT      = 33;
const HAND_COUNT      = 21;
const LANDMARK_DIM    = 233;

// MediaPipe Pose landmark indices
const IDX_NOSE        = 0;
const IDX_L_SHOULDER  = 11;
const IDX_R_SHOULDER  = 12;
const IDX_L_WRIST     = 15;
const IDX_R_WRIST     = 16;
const IDX_L_HIP       = 23;
const IDX_R_HIP       = 24;
const IDX_MCP_MIDDLE  = 9;   // wrist → middle MCP, used for palm scale


function _zeroArray(n) {
  return new Float32Array(n).fill(0);
}


function _lmToArray(landmarks, count) {
  // MediaPipe JS landmark list → Float32Array of length count*3
  // Zero-pads if fewer landmarks than expected.
  const out = new Float32Array(count * 3);
  if (!landmarks) return out;
  const n = Math.min(landmarks.length, count);
  for (let i = 0; i < n; i++) {
    out[i * 3]     = landmarks[i].x;
    out[i * 3 + 1] = landmarks[i].y;
    out[i * 3 + 2] = landmarks[i].z;
  }
  return out;
}


function _normalizeHand(handArr) {
  // handArr: Float32Array of length 63 (21 × 3, x/y/z interleaved)
  // Returns wrist-relative, palm-scale-normalised Float32Array(63)
  const out    = new Float32Array(63);
  const wrist  = [handArr[0], handArr[1], handArr[2]];

  // Subtract wrist
  for (let i = 0; i < 21; i++) {
    out[i * 3]     = handArr[i * 3]     - wrist[0];
    out[i * 3 + 1] = handArr[i * 3 + 1] - wrist[1];
    out[i * 3 + 2] = handArr[i * 3 + 2] - wrist[2];
  }

  // Palm scale = distance from wrist to middle MCP (landmark 9)
  const mx = out[IDX_MCP_MIDDLE * 3];
  const my = out[IDX_MCP_MIDDLE * 3 + 1];
  const mz = out[IDX_MCP_MIDDLE * 3 + 2];
  const scale = Math.sqrt(mx * mx + my * my + mz * mz);

  if (scale > 1e-6) {
    for (let i = 0; i < 63; i++) out[i] /= scale;
  }
  return out;
}


function _bodyAnchor(wristY, nose, lShoulder, rShoulder, midHip, shoulderWidth) {
  // Returns Float32Array(4) — vertical offsets from 4 body landmarks,
  // shoulder-width normalised. Matches Python body_anchor() exactly.
  const scale = shoulderWidth > 1e-6 ? shoulderWidth : 1.0;
  return new Float32Array([
    (wristY - nose[1])       / scale,
    (wristY - lShoulder[1])  / scale,
    (wristY - rShoulder[1])  / scale,
    (wristY - midHip[1])     / scale,
  ]);
}


/**
 * extractLandmarkVector(results)
 *
 * Call this inside your Holistic onResults callback.
 * Returns Float32Array(233) if pose is detected, or null if not.
 *
 * @param {object} results  — MediaPipe Holistic JS results object
 * @returns {Float32Array|null}
 */
export function extractLandmarkVector(results) {
  if (!results.poseLandmarks) return null;   // pose is the reliability anchor

  // ── Pose (33 × 3 = 99) ───────────────────────────────────────────────────
  const poseArr = _lmToArray(results.poseLandmarks, POSE_COUNT);

  // ── Body reference points ─────────────────────────────────────────────────
  const nose      = [poseArr[IDX_NOSE      * 3], poseArr[IDX_NOSE      * 3 + 1], poseArr[IDX_NOSE      * 3 + 2]];
  const lShoulder = [poseArr[IDX_L_SHOULDER * 3], poseArr[IDX_L_SHOULDER * 3 + 1], poseArr[IDX_L_SHOULDER * 3 + 2]];
  const rShoulder = [poseArr[IDX_R_SHOULDER * 3], poseArr[IDX_R_SHOULDER * 3 + 1], poseArr[IDX_R_SHOULDER * 3 + 2]];
  const lHip      = [poseArr[IDX_L_HIP      * 3], poseArr[IDX_L_HIP      * 3 + 1], poseArr[IDX_L_HIP      * 3 + 2]];
  const rHip      = [poseArr[IDX_R_HIP      * 3], poseArr[IDX_R_HIP      * 3 + 1], poseArr[IDX_R_HIP      * 3 + 2]];
  const midHip    = [(lHip[0] + rHip[0]) / 2, (lHip[1] + rHip[1]) / 2, (lHip[2] + rHip[2]) / 2];

  const dx           = lShoulder[0] - rShoulder[0];
  const dy           = lShoulder[1] - rShoulder[1];
  const shoulderWidth = Math.sqrt(dx * dx + dy * dy);

  // ── Hands (21 × 3 = 63 each) ─────────────────────────────────────────────
  const lHandArr = _lmToArray(results.leftHandLandmarks,  HAND_COUNT);
  const rHandArr = _lmToArray(results.rightHandLandmarks, HAND_COUNT);

  const lHandNorm = _normalizeHand(lHandArr);   // (63,)
  const rHandNorm = _normalizeHand(rHandArr);   // (63,)

  // ── Body anchors ──────────────────────────────────────────────────────────
  // Python uses pose[15] = left wrist, pose[16] = right wrist
  const lWristY = poseArr[IDX_L_WRIST * 3 + 1];
  const rWristY = poseArr[IDX_R_WRIST * 3 + 1];

  const lAnchor = _bodyAnchor(lWristY, nose, lShoulder, rShoulder, midHip, shoulderWidth);
  const rAnchor = _bodyAnchor(rWristY, nose, lShoulder, rShoulder, midHip, shoulderWidth);

  // ── Assemble 233-dim vector ───────────────────────────────────────────────
  const out = new Float32Array(LANDMARK_DIM);
  let offset = 0;

  out.set(poseArr,   offset); offset += 99;
  out.set(lHandNorm, offset); offset += 63;
  out.set(rHandNorm, offset); offset += 63;
  out.set(lAnchor,   offset); offset += 4;
  out.set(rAnchor,   offset); offset += 4;
  // offset === 233 ✓

  return out;
}


/**
 * Quick sanity check — call once on first valid frame.
 * Logs dimension and value range to console.
 */
export function verifyExtractor(vec) {
  if (!vec) { console.warn("[MediBridge] verifyExtractor: null vector"); return; }
  if (vec.length !== LANDMARK_DIM) {
    console.error(`[MediBridge] Expected ${LANDMARK_DIM} dims, got ${vec.length}`);
    return;
  }
  const poseX  = Array.from(vec.slice(0, 99)).filter((_, i) => i % 3 === 0);
  const min    = Math.min(...poseX).toFixed(3);
  const max    = Math.max(...poseX).toFixed(3);
  console.log(`[MediBridge] Extractor OK — 233 dims. Pose x range: ${min} – ${max}`);
  // Expected: roughly 0.3 – 0.7 for a centred signer (matches your Python check)
}