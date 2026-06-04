"""
medibridge_record.py  —  MediBridge v2  Record → Segment → Predict → Speak
===========================================================================
Controls (click the camera window first)
-----------------------------------------
    S  — start recording
    Q  — stop recording and process
    C  — cancel / clear without processing
    X  — exit

Flow
----
  Live webcam → HolisticLandmarker → frame buffer
  On Q: buffer → sign segmenter → per-segment Transformer prediction
       → sentence string → pyttsx3 TTS → saved .txt
"""

import matplotlib
matplotlib.use('Agg')   # no display needed for matplotlib

import cv2
import pickle
import numpy as np
import mediapipe as mp
import tensorflow as tf
import pyttsx3
import datetime
import os
from scipy.interpolate import interp1d
from mediapipe.tasks import python as mptasks
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import HolisticLandmarkerOptions, RunningMode

# ── Load model + label encoder ────────────────────────────────────────────────
print('Loading model...')
model = tf.keras.models.load_model('best_model.keras')

with open('label_encoder.pkl', 'rb') as f:
    le = pickle.load(f)

# Read shape directly from model — no hardcoded constants
TARGET_FRAMES  = int(model.input_shape[1])   # e.g. 45
TOTAL_FEAT_DIM = int(model.input_shape[2])   # e.g. 466
LANDMARK_DIM   = TOTAL_FEAT_DIM // 2         # 233  (raw; other half is velocity)

print(f'Model loaded.  Input shape : {model.input_shape}')
print(f'  TARGET_FRAMES  = {TARGET_FRAMES}')
print(f'  LANDMARK_DIM   = {LANDMARK_DIM}')
print(f'Classes: {[str(c) for c in le.classes_]}')

# ── HolisticLandmarker ────────────────────────────────────────────────────────
_base = mptasks.BaseOptions(model_asset_path='holistic_landmarker.task')
_opts = HolisticLandmarkerOptions(
    base_options=_base,
    running_mode=RunningMode.IMAGE,
    min_pose_detection_confidence=0.5,
    min_pose_landmarks_confidence=0.5,
    min_hand_landmarks_confidence=0.5,
    output_segmentation_mask=False,   # prevents RET_CHECK crash on size changes
)
detector = vision.HolisticLandmarker.create_from_options(_opts)

# ── TTS engine ────────────────────────────────────────────────────────────────
tts = pyttsx3.init()
tts.setProperty('rate', 150)

# ── Config ────────────────────────────────────────────────────────────────────
MIN_VALID_FRAMES     = 5     # discard segments shorter than this
NO_HAND_GAP_FRAMES   = 6     # consecutive no-pose frames that split two signs
CONFIDENCE_THRESHOLD = 0.60  # segments below this are shown as [word?]
RECORDINGS_DIR       = './recordings'

os.makedirs(RECORDINGS_DIR, exist_ok=True)


# ── Feature helpers — identical to extract_keypoints.py ──────────────────────

def _lm_list_to_array(lm_list, n: int) -> np.ndarray:
    """List[NormalizedLandmark] → (n, 3) float32.  All-zeros if empty."""
    if not lm_list:
        return np.zeros((n, 3), dtype=np.float32)
    pts = np.array([[lm.x, lm.y, lm.z] for lm in lm_list], dtype=np.float32)
    if len(pts) < n:
        pts = np.vstack([pts, np.zeros((n - len(pts), 3), dtype=np.float32)])
    return pts[:n]


def _normalize_hand(hand_pts: np.ndarray) -> np.ndarray:
    """Wrist-relative, palm-scale-normalised → (63,) float32."""
    wrist  = hand_pts[0].copy()
    coords = hand_pts - wrist
    scale  = np.linalg.norm(coords[9])
    if scale > 1e-6:
        coords /= scale
    return coords.flatten()


def extract_frame_features(result) -> np.ndarray:
    """
    233-dim feature vector — layout identical to extract_keypoints.py:
      [  0: 99]  pose landmarks,       33 × 3
      [ 99:162]  left-hand normalised, 21 × 3
      [162:225]  right-hand normalised, 21 × 3
      [225:229]  left-wrist  body anchor, 4 values
      [229:233]  right-wrist body anchor, 4 values
    """
    pose  = _lm_list_to_array(result.pose_landmarks,       33)
    lhand = _lm_list_to_array(result.left_hand_landmarks,  21)
    rhand = _lm_list_to_array(result.right_hand_landmarks, 21)

    lhand_norm = _normalize_hand(lhand)
    rhand_norm = _normalize_hand(rhand)

    nose       = pose[0]
    l_shoulder = pose[11]
    r_shoulder = pose[12]
    mid_hip    = (pose[23] + pose[24]) / 2.0
    shoulder_w = np.linalg.norm(l_shoulder - r_shoulder)
    scale      = shoulder_w if shoulder_w > 1e-6 else 1.0

    def body_anchor(wrist_pos):
        return np.array([
            (wrist_pos[1] - nose[1])       / scale,
            (wrist_pos[1] - l_shoulder[1]) / scale,
            (wrist_pos[1] - r_shoulder[1]) / scale,
            (wrist_pos[1] - mid_hip[1])    / scale,
        ], dtype=np.float32)

    return np.concatenate([
        pose.flatten(),
        lhand_norm,
        rhand_norm,
        body_anchor(pose[15]),
        body_anchor(pose[16]),
    ]).astype(np.float32)   # 233 values


def is_frame_reliable(result) -> bool:
    """True when pose is detected — pose is the spatial anchor for all features."""
    return bool(result.pose_landmarks)


# ── Sequence helpers ──────────────────────────────────────────────────────────

def resample_sequence(frames, target=TARGET_FRAMES) -> np.ndarray:
    """Resample variable-length list of feature vectors to exactly `target` frames."""
    arr = np.array(frames, dtype=np.float32)
    n   = len(arr)
    if n == 0:
        return np.zeros((target, LANDMARK_DIM), dtype=np.float32)
    if n == 1:
        return np.tile(arr[0], (target, 1))
    x_old = np.linspace(0, 1, n)
    x_new = np.linspace(0, 1, target)
    return interp1d(x_old, arr, axis=0)(x_new).astype(np.float32)


def add_velocity(seq: np.ndarray) -> np.ndarray:
    """(T, F) → (T, 2F) — prepend-diff matches extract_keypoints.py."""
    vel = np.diff(seq, axis=0, prepend=seq[:1])
    return np.concatenate([seq, vel], axis=-1).astype(np.float32)


# ── Sign segmentation ─────────────────────────────────────────────────────────

def segment_recording(frame_data: list) -> list:
    """
    Split the frame buffer into individual sign segments.
    A segment boundary is triggered by NO_HAND_GAP_FRAMES consecutive
    frames where pose was not reliably detected.
    """
    segments      = []
    current_seg   = []
    no_hand_count = 0

    for fd in frame_data:
        if fd['reliable'] and fd['keypoints'] is not None:
            no_hand_count = 0
            current_seg.append(fd['keypoints'])
        else:
            no_hand_count += 1
            if no_hand_count >= NO_HAND_GAP_FRAMES and current_seg:
                segments.append(current_seg)
                current_seg   = []
                no_hand_count = 0

    if current_seg:
        segments.append(current_seg)

    return segments


# ── Prediction ────────────────────────────────────────────────────────────────

def predict_segment(segment: list) -> tuple:
    """(list of 233-dim vectors) → (label: str, confidence: float)."""
    if len(segment) < MIN_VALID_FRAMES:
        return '[too short]', 0.0

    seq    = resample_sequence(segment)          # (TARGET_FRAMES, LANDMARK_DIM)
    seq    = add_velocity(seq)                   # (TARGET_FRAMES, TOTAL_FEAT_DIM)
    tensor = seq[np.newaxis, ...]                # (1, TARGET_FRAMES, TOTAL_FEAT_DIM)

    proba      = model.predict(tensor, verbose=0)[0]
    idx        = int(np.argmax(proba))
    confidence = float(proba[idx])
    label      = str(le.inverse_transform([idx])[0])   # str() strips np.str_

    return label, confidence


def process_recording(frame_data: list) -> str:
    """Full pipeline: frame buffer → segments → predictions → sentence string."""
    print(f'\nProcessing {len(frame_data)} recorded frames...')
    segments = segment_recording(frame_data)
    print(f'Detected {len(segments)} sign segment(s).')

    if not segments:
        print('No signs detected — were you visible in the frame during recording?')
        return ''

    words = []
    print('\n── Sign-by-sign results ──────────────────────────────────────')
    for i, seg in enumerate(segments, 1):
        label, conf = predict_segment(seg)
        flag = '✓' if conf >= CONFIDENCE_THRESHOLD else '?'
        print(f'  Sign {i:>2} : {label:<20} {conf * 100:.0f}%  [{flag}]  '
              f'({len(seg)} frames)')
        words.append(label if conf >= CONFIDENCE_THRESHOLD else f'[{label}?]')

    sentence = ' '.join(words)
    print(f'\nSentence : {sentence}')
    return sentence


# ── TTS + save ────────────────────────────────────────────────────────────────

def speak_and_save(sentence: str) -> None:
    if not sentence:
        return
    tts.say(sentence)
    tts.runAndWait()

    ts   = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    path = os.path.join(RECORDINGS_DIR, f'sentence_{ts}.txt')
    with open(path, 'w', encoding='utf-8') as f:
        f.write(sentence + '\n')
    print(f'Saved → {path}')


# ── UI overlay ────────────────────────────────────────────────────────────────

def draw_ui(frame, recording: bool, n_frames: int, status: str):
    display    = frame.copy()
    h, w       = display.shape[:2]
    bar_colour = (0, 0, 180) if recording else (40, 40, 40)

    # Top bar
    cv2.rectangle(display, (0, 0), (w, 55), bar_colour, -1)
    label = f'● REC  {n_frames} frames' if recording else 'Press S to start recording'
    cv2.putText(display, label, (10, 38),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

    # Bottom bar
    cv2.rectangle(display, (0, h - 80), (w, h), (20, 20, 20), -1)
    cv2.putText(display, 'S=start  Q=stop+process  C=cancel  X=exit',
                (10, h - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
    short_status = status[:55] if len(status) > 55 else status
    cv2.putText(display, short_status, (10, h - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)

    return display


# ── Main loop ─────────────────────────────────────────────────────────────────

cap        = cv2.VideoCapture(0)
recording  = False
frame_data = []
status     = 'Ready'

print('\nMediBridge — Record Mode')
print('Click the camera window first, then use the keys shown on screen.\n')

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Fixed resolution prevents HolisticLandmarker segmentation crash
    frame     = cv2.resize(frame, (640, 480))
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_img    = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    result    = detector.detect(mp_img)
    reliable  = is_frame_reliable(result)

    if recording:
        frame_data.append({
            'reliable' : reliable,
            'keypoints': extract_frame_features(result).tolist() if reliable else None,
        })
        # Indicator dot: green = pose detected, red = not detected
        dot_colour = (0, 255, 0) if reliable else (0, 0, 255)
        cv2.circle(frame, (frame.shape[1] - 30, 30), 12, dot_colour, -1)

    cv2.imshow('MediBridge — Record Mode',
               draw_ui(frame, recording, len(frame_data), status))

    key      = cv2.waitKey(25) & 0xFF
    key_char = chr(key).lower() if 0 < key < 128 else ''

    if key_char == 's' and not recording:
        recording  = True
        frame_data = []
        status     = 'Recording... (press Q to stop)'
        print('Recording started.')

    elif key_char == 'q' and recording:
        recording = False
        status    = 'Processing...'
        print('Recording stopped.')
        cv2.imshow('MediBridge — Record Mode',
                   draw_ui(frame, False, len(frame_data), status))
        cv2.waitKey(1)

        sentence   = process_recording(frame_data)
        frame_data = []
        status     = (sentence[:55] if len(sentence) > 55 else sentence) \
                     if sentence else 'No sentence detected. Try again.'

        if sentence:
            print('Speaking...')
            speak_and_save(sentence)

    elif key_char == 'c':
        recording  = False
        frame_data = []
        status     = 'Cleared. Press S to start.'
        print('Recording cleared.')

    elif key_char == 'x':
        break

# ── Cleanup ───────────────────────────────────────────────────────────────────
detector.close()
cap.release()
cv2.destroyAllWindows()
print('Exited.')