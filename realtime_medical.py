"""
realtime_medical.py  —  MediBridge v2  (MediaPipe Tasks API ≥ 0.10)
====================================================================
Reads SEQ_LEN and FEATURE_DIM directly from model.input_shape —
no model_meta.pkl file needed.

Feature extraction is byte-for-byte identical to extract_keypoints.py.

Required files (same directory as this script):
  best_model.keras          ← from train_model.py
  label_encoder.pkl         ← from train_model.py
  holistic_landmarker.task  ← downloaded from MediaPipe model hub
"""

import cv2
import pickle
import numpy as np
import tensorflow as tf
from collections import deque
from scipy.interpolate import interp1d
import mediapipe as mp
from mediapipe.tasks import python as mptasks
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import HolisticLandmarkerOptions, RunningMode

# ── Load model ────────────────────────────────────────────────────────────────
model = tf.keras.models.load_model('best_model.keras')

with open('label_encoder.pkl', 'rb') as f:
    le = pickle.load(f)

# Read dimensions from the model itself — no external metadata file needed
SEQ_LEN     = int(model.input_shape[1])   # e.g. 45
FEATURE_DIM = int(model.input_shape[2])   # e.g. 466
RAW_DIM     = FEATURE_DIM // 2            # 233

print(f'Model loaded   — seq_len={SEQ_LEN}, feature_dim={FEATURE_DIM}')
print(f'Classes        — {list(le.classes_)}')

# ── HolisticLandmarker ────────────────────────────────────────────────────────
HOLISTIC_MODEL_PATH = 'holistic_landmarker.task'

base_opts = mptasks.BaseOptions(model_asset_path=HOLISTIC_MODEL_PATH)
hol_opts  = HolisticLandmarkerOptions(
    base_options=base_opts,
    running_mode=RunningMode.IMAGE,
    min_pose_detection_confidence=0.5,
    min_pose_landmarks_confidence=0.5,
    min_hand_landmarks_confidence=0.5,
    output_segmentation_mask=False,   # prevents RET_CHECK crash on size changes
)
holistic = vision.HolisticLandmarker.create_from_options(hol_opts)

# ── Inference constants ───────────────────────────────────────────────────────
CONFIDENCE_THRESHOLD   = 0.70
STABLE_FRAME_THRESHOLD = 5
NO_HAND_COMMIT_FRAMES  = 8

# ── State ─────────────────────────────────────────────────────────────────────
cap                = cv2.VideoCapture(0)
prediction         = 'Waiting...'
sentence           = []
gesture_history    = deque(maxlen=SEQ_LEN)   # raw RAW_DIM-dim frames
prediction_history = deque(maxlen=10)
stable_word        = None
stable_word_frames = 0
no_hand_frames     = 0

print('MediBridge running — Press Q to quit, C to clear sentence.')


# ── Feature helpers — MUST be identical to extract_keypoints.py ───────────────

def _lm_list_to_array(lm_list, n):
    if not lm_list:
        return np.zeros((n, 3), dtype=np.float32)
    pts = np.array([[lm.x, lm.y, lm.z] for lm in lm_list], dtype=np.float32)
    if len(pts) < n:
        pts = np.vstack([pts, np.zeros((n - len(pts), 3), dtype=np.float32)])
    return pts[:n]


def _normalize_hand(hand_pts):
    wrist  = hand_pts[0].copy()
    coords = hand_pts - wrist
    scale  = np.linalg.norm(coords[9])
    if scale > 1e-6:
        coords /= scale
    return coords.flatten()


def extract_frame_features(result):
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


def build_input_tensor(history):
    """
    history : deque of RAW_DIM-dim frames (variable length, up to SEQ_LEN)
    Returns : (1, SEQ_LEN, FEATURE_DIM) tensor ready for model.predict()
    """
    arr = np.array(list(history), dtype=np.float32)   # (n, RAW_DIM)
    n   = len(arr)

    # Resample to exactly SEQ_LEN
    if n != SEQ_LEN:
        x_old = np.linspace(0, 1, n)
        x_new = np.linspace(0, 1, SEQ_LEN)
        arr   = interp1d(x_old, arr, axis=0)(x_new).astype(np.float32)

    # Velocity — identical to add_velocity() in extract_keypoints.py
    vel = np.diff(arr, axis=0, prepend=arr[:1])
    seq = np.concatenate([arr, vel], axis=-1)           # (SEQ_LEN, FEATURE_DIM)
    return seq[np.newaxis, ...]                         # (1, SEQ_LEN, FEATURE_DIM)


def sentence_to_text(words):
    return ' '.join(words).strip()


def save_sentence(text):
    if not text:
        return
    with open('recognized_sentence.txt', 'a', encoding='utf-8') as f:
        f.write(text + '\n')


# ── Main loop ─────────────────────────────────────────────────────────────────
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Keep a consistent resolution so Holistic's internal state stays stable
    frame = cv2.resize(frame, (640, 480))

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_img    = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    result    = holistic.detect(mp_img)

    pose_detected = bool(result.pose_landmarks)

    if pose_detected:
        no_hand_frames = 0
        gesture_history.append(extract_frame_features(result))

        # Wait for at least half a window before predicting
        if len(gesture_history) < SEQ_LEN // 2:
            prediction = f'Collecting... {len(gesture_history)}/{SEQ_LEN}'
            prediction_history.clear()
        else:
            inp   = build_input_tensor(gesture_history)    # (1, SEQ_LEN, FEATURE_DIM)
            probs = model.predict(inp, verbose=0)[0]       # (num_classes,)
            idx   = int(np.argmax(probs))
            conf  = float(probs[idx])
            word  = str(le.inverse_transform([idx])[0])   # str() strips np.str_

            prediction = f'{word.upper()}  {conf * 100:.0f}%'
            prediction_history.append(word)

            if len(prediction_history) >= STABLE_FRAME_THRESHOLD and conf >= CONFIDENCE_THRESHOLD:
                counts   = {}
                for w in prediction_history:
                    counts[w] = counts.get(w, 0) + 1
                top_word  = max(counts, key=counts.get)
                top_count = counts[top_word]

                if top_count >= STABLE_FRAME_THRESHOLD:
                    if top_word == stable_word:
                        stable_word_frames += 1
                    else:
                        stable_word        = top_word
                        stable_word_frames = 1

    else:
        prediction     = 'No body detected'
        no_hand_frames += 1
        gesture_history.clear()

        if no_hand_frames >= NO_HAND_COMMIT_FRAMES and stable_word:
            sentence.append(stable_word)
            print(f'Word committed : {stable_word}')
            stable_word        = None
            stable_word_frames = 0
            prediction_history.clear()

    # ── Display ───────────────────────────────────────────────────────────────
    display = frame.copy()

    # Top bar — current prediction
    cv2.rectangle(display, (0, 0), (640, 70), (0, 0, 0), -1)
    cv2.putText(display, prediction, (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

    # Collection progress bar (cyan strip under the top bar)
    filled = int(640 * min(len(gesture_history), SEQ_LEN) / SEQ_LEN)
    cv2.rectangle(display, (0, 68), (filled, 75), (0, 200, 255), -1)

    # Bottom bar — accumulated sentence
    cv2.rectangle(display,
                  (0, display.shape[0] - 60),
                  (display.shape[1], display.shape[0]),
                  (0, 0, 0), -1)
    cv2.putText(display, sentence_to_text(sentence),
                (10, display.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

    cv2.imshow('MediBridge — Medical Sign Language', display)

    key = cv2.waitKey(25)
    if key == ord('q'):
        break
    if key == ord('c'):
        sentence.clear()
        prediction_history.clear()
        gesture_history.clear()
        stable_word        = None
        stable_word_frames = 0
        no_hand_frames     = 0
        print('Sentence cleared.')

# ── Cleanup ───────────────────────────────────────────────────────────────────
holistic.close()
cap.release()
cv2.destroyAllWindows()

final = sentence_to_text(sentence)
save_sentence(final)
print(f'Final sentence : {final}')