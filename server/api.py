"""
api.py  —  MediBridge FastAPI Backend
======================================
Receives 233-dim landmark vectors from the browser (MediaPipe Holistic JS),
maintains per-session rolling buffers, segments signs, and returns predictions.

Endpoints
---------
POST /predict          — receive one landmark frame, get prediction if ready
POST /session/reset    — clear a session's buffer
GET  /health           — liveness check

Run with:
    uvicorn api:app --host 0.0.0.0 --port 8000 --reload
"""

import pickle
import uuid
import time
import numpy as np
import tensorflow as tf
from collections import deque
from scipy.interpolate import interp1d

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional

# ── Load model + label encoder ────────────────────────────────────────────────
print("Loading model...")
import os
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

model = tf.keras.models.load_model(os.path.join(_ROOT, "model.keras"))
with open(os.path.join(_ROOT, "label_encoder.pkl"), "rb") as f:
    le = pickle.load(f)
# model = tf.keras.models.load_model("model.keras")
# with open("label_encoder.pkl", "rb") as f:
#     le = pickle.load(f)

_input_shape   = model.input_shape       # (None, 45, 466)
TARGET_FRAMES  = _input_shape[1]         # 45
TOTAL_FEAT_DIM = _input_shape[2]         # 466
LANDMARK_DIM   = TOTAL_FEAT_DIM // 2    # 233

print(f"Model ready. Input: {_input_shape}")
print(f"Classes: {list(le.classes_)}")

# ── Config ────────────────────────────────────────────────────────────────────
CONFIDENCE_THRESHOLD  = 0.60
MIN_VALID_FRAMES      = 5
NO_HAND_GAP_FRAMES    = 6     # consecutive empty frames that close a sign segment
MAX_BUFFER_FRAMES     = 150   # ~5s at 30fps — safety cap per session
SESSION_TTL_SECONDS   = 300   # sessions idle longer than this are purged
MAX_SESSIONS          = 200   # guard against memory exhaustion

# ── Session store ─────────────────────────────────────────────────────────────
# Each session tracks:
#   buffer        — rolling deque of 233-dim landmark vectors (or None = no hand)
#   no_hand_count — consecutive frames with no hand detected
#   last_seen     — timestamp for TTL-based cleanup
sessions: dict[str, dict] = {}


def get_or_create_session(session_id: str) -> dict:
    if session_id not in sessions:
        if len(sessions) >= MAX_SESSIONS:
            _evict_oldest_session()
        sessions[session_id] = {
            "buffer":        deque(maxlen=MAX_BUFFER_FRAMES),
            "no_hand_count": 0,
            "last_seen":     time.time(),
        }
    else:
        sessions[session_id]["last_seen"] = time.time()
    return sessions[session_id]


def _evict_oldest_session():
    """Remove the session that has been idle the longest."""
    oldest = min(sessions, key=lambda k: sessions[k]["last_seen"])
    del sessions[oldest]


def _purge_stale_sessions():
    """Remove sessions idle beyond SESSION_TTL_SECONDS."""
    now = time.time()
    stale = [k for k, v in sessions.items()
             if now - v["last_seen"] > SESSION_TTL_SECONDS]
    for k in stale:
        del sessions[k]


# ── Feature helpers ───────────────────────────────────────────────────────────

def resample_sequence(frames: list, target: int = TARGET_FRAMES) -> np.ndarray:
    arr = np.array(frames, dtype=np.float32)
    n = len(arr)
    if n == 0:
        return np.zeros((target, LANDMARK_DIM), dtype=np.float32)
    if n == 1:
        return np.tile(arr[0], (target, 1))
    x_old = np.linspace(0, 1, n)
    x_new = np.linspace(0, 1, target)
    return interp1d(x_old, arr, axis=0)(x_new).astype(np.float32)


def add_velocity(seq: np.ndarray) -> np.ndarray:
    """(T, 233) → (T, 466) by prepending zero-row velocity diff."""
    vel = np.diff(seq, axis=0, prepend=seq[:1])
    return np.concatenate([seq, vel], axis=-1).astype(np.float32)


def predict_segment(segment: list) -> dict:
    """List of 233-dim vectors → {label, confidence, all_probs}."""
    if len(segment) < MIN_VALID_FRAMES:
        return {"label": None, "confidence": 0.0, "reason": "too_short"}

    seq    = resample_sequence(segment)           # (45, 233)
    seq    = add_velocity(seq)                    # (45, 466)
    tensor = seq[np.newaxis, ...]                 # (1, 45, 466)

    proba      = model.predict(tensor, verbose=0)[0]
    class_idx  = int(np.argmax(proba))
    confidence = float(proba[class_idx])
    label      = le.inverse_transform([class_idx])[0]

    return {
        "label":      label,
        "confidence": round(confidence, 4),
        "above_threshold": confidence >= CONFIDENCE_THRESHOLD,
        "all_probs":  {
            cls: round(float(p), 4)
            for cls, p in zip(le.classes_, proba)
        },
    }


# ── Request / response models ─────────────────────────────────────────────────

class LandmarkFrame(BaseModel):
    session_id: str = Field(..., description="UUID identifying the user session")
    landmarks: Optional[list[float]] = Field(
        None,
        description="233-dim landmark vector from MediaPipe Holistic JS. "
                    "Pass null / omit when no hand is detected in this frame."
    )
    timestamp_ms: Optional[int] = Field(
        None, description="Browser timestamp (ms) — logged but not used for inference"
    )


class PredictionResponse(BaseModel):
    session_id:   str
    prediction:   Optional[dict]  # populated when a sign segment closes
    buffer_depth: int             # current number of buffered frames
    status:       str             # 'buffering' | 'predicted' | 'gap_detected' | 'too_short'


class ResetResponse(BaseModel):
    session_id: str
    cleared_frames: int
    status: str


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="MediBridge Sign Language API",
    description="Receives MediaPipe Holistic landmark vectors and returns sign predictions.",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # tighten to your frontend domain in production
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {
        "status":        "ok",
        "model_classes": len(le.classes_),
        "active_sessions": len(sessions),
        "target_frames": TARGET_FRAMES,
        "landmark_dim":  LANDMARK_DIM,
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(frame: LandmarkFrame):
    # ── Purge stale sessions periodically (every ~50 requests) ───────────────
    if len(sessions) > 0 and hash(frame.session_id) % 50 == 0:
        _purge_stale_sessions()

    session = get_or_create_session(frame.session_id)
    buf     = session["buffer"]

    # ── No hand detected this frame ───────────────────────────────────────────
    if frame.landmarks is None:
        session["no_hand_count"] += 1

        # Gap threshold reached → close the current segment and predict
        if session["no_hand_count"] == NO_HAND_GAP_FRAMES and len(buf) > 0:
            segment = list(buf)
            buf.clear()
            session["no_hand_count"] = 0
            result = predict_segment(segment)
            return PredictionResponse(
                session_id   = frame.session_id,
                prediction   = result,
                buffer_depth = 0,
                status       = "predicted",
            )

        return PredictionResponse(
            session_id   = frame.session_id,
            prediction   = None,
            buffer_depth = len(buf),
            status       = "gap_detected",
        )

    # ── Hand detected — validate landmark vector ──────────────────────────────
    if len(frame.landmarks) != LANDMARK_DIM:
        raise HTTPException(
            status_code=422,
            detail=f"Expected {LANDMARK_DIM} landmark values, got {len(frame.landmarks)}. "
                   f"Ensure MediaPipe Holistic JS extracts the full holistic feature vector."
        )

    session["no_hand_count"] = 0
    buf.append(np.array(frame.landmarks, dtype=np.float32))

    return PredictionResponse(
        session_id   = frame.session_id,
        prediction   = None,
        buffer_depth = len(buf),
        status       = "buffering",
    )


@app.post("/session/reset", response_model=ResetResponse)
def reset_session(body: dict):
    session_id = body.get("session_id")
    if not session_id:
        raise HTTPException(status_code=422, detail="session_id required")

    cleared = 0
    if session_id in sessions:
        cleared = len(sessions[session_id]["buffer"])
        del sessions[session_id]

    return ResetResponse(
        session_id    = session_id,
        cleared_frames = cleared,
        status        = "cleared",
    )


@app.get("/session/new")
def new_session():
    """Convenience endpoint — browser can request a fresh session UUID."""
    return {"session_id": str(uuid.uuid4())}