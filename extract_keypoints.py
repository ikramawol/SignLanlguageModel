import os
import cv2
import mediapipe as mp
import numpy as np
import pickle

from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

DATA_DIR = './video_data'
os.makedirs('./keypoint_data', exist_ok=True)

AUGMENTATIONS_PER_SAMPLE = 3
NOISE_STD = 0.01

# New MediaPipe API
base_options = mp_python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=2)
detector = vision.HandLandmarker.create_from_options(options)

data = []
labels = []


def mirror_keypoints(feature_vector):
    mirrored = feature_vector.copy()
    for index in range(0, len(mirrored), 3):
        mirrored[index] = 1.0 - mirrored[index]
    return mirrored


def jitter_keypoints(feature_vector):
    noisy = feature_vector.copy()
    noise = np.random.normal(0.0, NOISE_STD, size=noisy.shape)
    noisy = noisy + noise
    return np.clip(noisy, 0.0, 1.0)

for class_name in os.listdir(DATA_DIR):
    class_path = os.path.join(DATA_DIR, class_name)
    if not os.path.isdir(class_path):
        continue

    print(f'Processing: {class_name}')

    for video_file in os.listdir(class_path):
        video_path = os.path.join(class_path, video_file)
        cap = cv2.VideoCapture(video_path)
        video_keypoints = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            results = detector.detect(mp_image)

            frame_keypoints = []
            if results.hand_landmarks:
                for hand in results.hand_landmarks:
                    for lm in hand:
                        frame_keypoints.extend([lm.x, lm.y, lm.z])

            # Pad to fixed size (2 hands x 21 x 3 = 126)
            while len(frame_keypoints) < 126:
                frame_keypoints.append(0.0)

            video_keypoints.append(frame_keypoints[:126])

        cap.release()

        if video_keypoints:
            video_feature = np.mean(video_keypoints, axis=0)
            data.append(video_feature)
            labels.append(class_name)

            augmented_samples = [
                mirror_keypoints(video_feature),
                jitter_keypoints(video_feature),
                jitter_keypoints(mirror_keypoints(video_feature)),
            ]

            for augmented_feature in augmented_samples[:AUGMENTATIONS_PER_SAMPLE]:
                data.append(augmented_feature)
                labels.append(class_name)

            print(f'  {video_file} -> {len(video_keypoints)} frames')

with open('keypoint_data/data.pkl', 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)

print(f'\nDone! {len(data)} samples saved.')
print(f'Classes: {set(labels)}')