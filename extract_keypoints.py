"""
extract_keypoints.py  —  MediBridge v2  (MediaPipe Tasks API ≥ 0.10)
=====================================================================
Pipeline: HolisticLandmarker → body-relative features → (N, 45, 466)

Feature layout per frame (233 raw dims):
  [  0: 99]  pose landmarks, 33 × 3 (x, y, z)
  [ 99:162]  left-hand  wrist-relative + palm-scale-normalised, 21 × 3
  [162:225]  right-hand wrist-relative + palm-scale-normalised, 21 × 3
  [225:229]  left-wrist  body-relative anchor, 4 values
  [229:233]  right-wrist body-relative anchor, 4 values

After add_velocity(): 233 × 2 = 466 dims per frame → (N, 45, 466)

Required file: holistic_landmarker.task
  Download: https://storage.googleapis.com/mediapipe-models/
            holistic_landmarker/holistic_landmarker/float16/latest/
            holistic_landmarker.task

⚠ IMPORTANT — folder names must be lowercase
  Check that every folder in video_data/ is lowercase (e.g. 'nose' not 'Nose').
  The old dataset had a 'Nose' folder which added a spurious class.
"""

import os
import cv2
import mediapipe as mp
import numpy as np
import pickle
from scipy.interpolate import interp1d
from mediapipe.tasks import python as mptasks
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import HolisticLandmarkerOptions, RunningMode

# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR            = './video_data'
OUT_DIR             = './keypoint_data'
HOLISTIC_MODEL_PATH = 'holistic_landmarker.task'
TARGET_FRAMES       = 45        # 1.5 s at 30 fps — covers most signs fully
NOISE_STD           = 0.005
RAW_DIM             = 233
ROTATION_ANGLES     = [-15, -10, 10, 15]   # degrees; symmetric around 0
FEATURE_DIM         = RAW_DIM * 2   # 466 after velocity

os.makedirs(OUT_DIR, exist_ok=True)

# ── HolisticLandmarker ────────────────────────────────────────────────────────
base_opts = mptasks.BaseOptions(model_asset_path=HOLISTIC_MODEL_PATH)
hol_opts  = HolisticLandmarkerOptions(
    base_options=base_opts,
    running_mode=RunningMode.IMAGE,
    min_pose_detection_confidence=0.5,
    min_pose_landmarks_confidence=0.5,
    min_hand_landmarks_confidence=0.5,
    output_segmentation_mask=False,   # disables SegmentationSmoothingCalculator
                                      # which crashes on mixed-resolution frames
)
holistic = vision.HolisticLandmarker.create_from_options(hol_opts)

data, labels = [], []


# ── Feature helpers ───────────────────────────────────────────────────────────

def _lm_list_to_array(lm_list, n):
    """List[NormalizedLandmark] → (n, 3) float32.  All-zeros if empty."""
    if not lm_list:
        return np.zeros((n, 3), dtype=np.float32)
    pts = np.array([[lm.x, lm.y, lm.z] for lm in lm_list], dtype=np.float32)
    if len(pts) < n:
        pts = np.vstack([pts, np.zeros((n - len(pts), 3), dtype=np.float32)])
    return pts[:n]


def _normalize_hand(hand_pts):
    """Wrist-relative, palm-scale-normalised → (63,) float32."""
    wrist  = hand_pts[0].copy()
    coords = hand_pts - wrist
    scale  = np.linalg.norm(coords[9])   # wrist → middle MCP distance
    if scale > 1e-6:
        coords /= scale
    return coords.flatten()


def extract_frame_features(result):
    """Build 233-dim feature vector from one HolisticLandmarkerResult."""
    pose  = _lm_list_to_array(result.pose_landmarks,       33)  # (33,3)
    lhand = _lm_list_to_array(result.left_hand_landmarks,  21)  # (21,3)
    rhand = _lm_list_to_array(result.right_hand_landmarks, 21)  # (21,3)

    lhand_norm = _normalize_hand(lhand)   # (63,)
    rhand_norm = _normalize_hand(rhand)   # (63,)

    # Body reference points (MediaPipe Pose indices)
    nose       = pose[0]
    l_shoulder = pose[11]
    r_shoulder = pose[12]
    mid_hip    = (pose[23] + pose[24]) / 2.0

    shoulder_w = np.linalg.norm(l_shoulder - r_shoulder)
    scale      = shoulder_w if shoulder_w > 1e-6 else 1.0

    def body_anchor(wrist_pos):
        """Y-offset from 4 body landmarks, shoulder-width-normalised. (4,)"""
        return np.array([
            (wrist_pos[1] - nose[1])       / scale,
            (wrist_pos[1] - l_shoulder[1]) / scale,
            (wrist_pos[1] - r_shoulder[1]) / scale,
            (wrist_pos[1] - mid_hip[1])    / scale,
        ], dtype=np.float32)

    feat = np.concatenate([
        pose.flatten(),        #  99
        lhand_norm,            #  63
        rhand_norm,            #  63
        body_anchor(pose[15]), #   4  (left  wrist, pose lm 15)
        body_anchor(pose[16]), #   4  (right wrist, pose lm 16)
    ])                         # 233 total
    return feat.astype(np.float32)


def add_velocity(seq):
    """(T, D) → (T, 2D) by prepending row-diff. Shape: (45,233)→(45,466)."""
    vel = np.diff(seq, axis=0, prepend=seq[:1])
    return np.concatenate([seq, vel], axis=-1).astype(np.float32)


def resample_sequence(frames, target=TARGET_FRAMES):
    """Resample a variable-length list of feature vectors to exactly `target` frames."""
    arr = np.array(frames, dtype=np.float32)
    n   = len(arr)
    if n == 0:
        return np.zeros((target, RAW_DIM), dtype=np.float32)
    if n == 1:
        return np.tile(arr, (target, 1))
    x_old = np.linspace(0, 1, n)
    x_new = np.linspace(0, 1, target)
    return interp1d(x_old, arr, axis=0)(x_new).astype(np.float32)

def rotate_sequence(seq, angle_deg):
    """
    Rotate the 2D (x, y) plane of pose landmarks and wrist body anchors
    by `angle_deg` degrees around the origin.

    Applies to:
      - Pose landmarks      [0:99]   → indices 0,1 of each (x,y,z) triplet
      - Body anchor dims    [225:233] → these are y-only offsets, not rotated
      - Hand-internal dims  [99:225]  → wrist-relative, NOT rotated

    Hand-internal landmarks are already expressed relative to their own wrist,
    so rotating the global frame does not change their internal geometry.
    Only pose and the global frame need rotation.
    """
    angle_rad = np.deg2rad(angle_deg)
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)

    rotated = seq.copy()   # (T, RAW_DIM) — operate on pre-velocity sequence

    # ── Rotate pose landmarks (33 × 3 = 99 values starting at index 0) ──────
    # Layout: [x0, y0, z0, x1, y1, z1, ..., x32, y32, z32]
    for i in range(33):
        base = i * 3
        x = rotated[:, base]
        y = rotated[:, base + 1]
        rotated[:, base]     =  cos_a * x - sin_a * y
        rotated[:, base + 1] =  sin_a * x + cos_a * y
        # z (depth) is left unchanged — rotation is in the image plane only

    # ── Body anchors [225:233] are shoulder-width-normalised y-offsets ───────
    # They encode vertical height relative to body landmarks — not affected
    # by in-plane rotation, so no change needed here.

    return rotated.astype(np.float32)

def mirror_sequence(seq):
    """Negate every x-component (idx 0, 3, 6, …) to mirror left↔right."""
    m = seq.copy()
    m[:, 0::3] *= -1
    return m


def time_stretch(seq, factor):
    """Compress/expand time by `factor` then resample back to TARGET_FRAMES."""
    n_new = max(2, int(len(seq) * factor))
    x_old = np.linspace(0, 1, len(seq))
    x_new = np.linspace(0, 1, n_new)
    stretched = interp1d(x_old, seq, axis=0)(x_new)
    return resample_sequence(list(stretched), TARGET_FRAMES)


def jitter_sequence(seq):
    return (seq + np.random.normal(0.0, NOISE_STD, size=seq.shape)).astype(np.float32)


# ── Main extraction loop ──────────────────────────────────────────────────────

for class_name in sorted(os.listdir(DATA_DIR)):
    class_path = os.path.join(DATA_DIR, class_name)
    if not os.path.isdir(class_path):
        continue

    # Warn about capitalisation issues (e.g. 'Nose' instead of 'nose')
    if class_name != class_name.lower():
        print(f'  ⚠ WARNING: folder "{class_name}" is not lowercase. '
              f'Rename to "{class_name.lower()}" to avoid duplicate classes.')

    print(f'\nProcessing class: {class_name}')

    for video_file in sorted(os.listdir(class_path)):
        video_path = os.path.join(class_path, video_file)
        cap        = cv2.VideoCapture(video_path)
        raw_frames = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Resize every frame to a fixed resolution so the Holistic
            # segmentation smoother never sees a size change mid-video.
            # Mixed-resolution frames cause a RET_CHECK crash otherwise.
            frame = cv2.resize(frame, (640, 480))

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img    = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            result    = holistic.detect(mp_img)

            if not result.pose_landmarks:
                # Pose not detected: insert zero frame to preserve timing
                raw_frames.append(np.zeros(RAW_DIM, dtype=np.float32))
            else:
                raw_frames.append(extract_frame_features(result))

        cap.release()

        if len(raw_frames) < 5:
            print(f'  {video_file} → skipped (< 5 frames)')
            continue

        seq      = resample_sequence(raw_frames)   # (45, 233)
        seq_full = add_velocity(seq)               # (45, 466)

        data.append(seq_full)
        labels.append(class_name)

        # augmented = [
        #     add_velocity(mirror_sequence(seq)),                    # mirror
        #     add_velocity(jitter_sequence(seq)),                    # jitter
        #     add_velocity(jitter_sequence(mirror_sequence(seq))),   # both
        #     add_velocity(time_stretch(seq, 0.8)),                  # faster
        #     add_velocity(time_stretch(seq, 1.2)),                  # slower
        # ]
        
        augmented = [
            add_velocity(mirror_sequence(seq)),                    # mirror
            add_velocity(jitter_sequence(seq)),                    # jitter
            add_velocity(jitter_sequence(mirror_sequence(seq))),   # both
            add_velocity(time_stretch(seq, 0.8)),                  # faster
            add_velocity(time_stretch(seq, 1.2)),                  # slower
            # ── Rotation augmentation ─────────────────────────────────────────────
            *[add_velocity(rotate_sequence(seq, a)) for a in ROTATION_ANGLES],
            # ── Rotation + mirror (covers off-angle signers on both sides) ────────
            *[add_velocity(mirror_sequence(rotate_sequence(seq, a)))
            for a in ROTATION_ANGLES],
        ]
        
        for aug in augmented:
            data.append(aug)
            labels.append(class_name)

        print(f'  {video_file} → {len(raw_frames)} frames → '
              f'{1 + len(augmented)} samples')

holistic.close()

# ── Verify final shapes before saving ────────────────────────────────────────
data_array = np.array(data, dtype=np.float32)
assert data_array.ndim == 3,                         "Expected 3D array (N, T, F)"
assert data_array.shape[1] == TARGET_FRAMES,         f"Expected T={TARGET_FRAMES}"
assert data_array.shape[2] == FEATURE_DIM,           f"Expected F={FEATURE_DIM}"

# ── Per-class sample count ────────────────────────────────────────────────────
unique, counts = np.unique(labels, return_counts=True)
print('\n── Per-class sample counts ──────────────────────────────────')
for cls, cnt in zip(unique, counts):
    bar = '█' * (cnt // 3)
    print(f'  {cls:<20} {cnt:>4}  {bar}')

print(f'\nDataset shape  : {data_array.shape}')
print(f'Feature dim    : {data_array.shape[2]}  (233 raw + 233 velocity)')
print(f'Total samples  : {len(data_array)}')

out_path = os.path.join(OUT_DIR, 'data.pkl')
with open(out_path, 'wb') as f:
    pickle.dump({
        'data':        data_array,
        'labels':      np.array(labels),
        'feature_dim': int(data_array.shape[2]),
        'seq_len':     int(data_array.shape[1]),
    }, f)
print(f'Saved → {out_path}')