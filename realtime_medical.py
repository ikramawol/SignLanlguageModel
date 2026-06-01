import cv2
import pickle
import numpy as np
import mediapipe as mp
from collections import deque
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

# Load model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# MediaPipe setup
base_options = mp_python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=2)
detector = vision.HandLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)
prediction = 'Waiting...'
sentence = []
prediction_history = deque(maxlen=10)
gesture_history = deque(maxlen=10)
stable_word = None
stable_word_frames = 0
no_hand_frames = 0

CONFIDENCE_THRESHOLD = 0.75
STABLE_FRAME_THRESHOLD = 5
NO_HAND_COMMIT_FRAMES = 3
GESTURE_WINDOW_FRAMES = 10

print('MediBridge running! Press Q to quit, C to clear sentence.')


def sentence_to_text(words):
    return ' '.join(words).strip()


def save_sentence(text):
    if not text:
        return

    with open('recognized_sentence.txt', 'a', encoding='utf-8') as f:
        f.write(text + '\n')


def build_gesture_feature(frame_keypoints_history):
    return np.mean(np.array(frame_keypoints_history), axis=0).tolist()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    results = detector.detect(mp_image)

    keypoints = []
    if results.hand_landmarks:
        for hand in results.hand_landmarks:
            for lm in hand:
                keypoints.extend([lm.x, lm.y, lm.z])

    # Pad
    while len(keypoints) < 126:
        keypoints.append(0.0)

    if results.hand_landmarks:
        no_hand_frames = 0
        gesture_history.append(keypoints[:126])

        if len(gesture_history) < GESTURE_WINDOW_FRAMES:
            prediction = 'Preprocessing gesture...'
            prediction_history.clear()
            continue

        gesture_feature = build_gesture_feature(gesture_history)

        # Predict
        pred = model.predict([gesture_feature])[0]
        confidence = max(model.predict_proba([gesture_feature])[0])
        prediction = f'{pred.upper()} ({confidence*100:.0f}%)'

        prediction_history.append(pred)

        if len(prediction_history) >= STABLE_FRAME_THRESHOLD:
            counts = {}
            for word in prediction_history:
                counts[word] = counts.get(word, 0) + 1

            candidate_word = max(counts, key=counts.get)
            candidate_count = counts[candidate_word]

            if candidate_count >= STABLE_FRAME_THRESHOLD and confidence >= CONFIDENCE_THRESHOLD:
                if candidate_word == stable_word:
                    stable_word_frames += 1
                else:
                    stable_word = candidate_word
                    stable_word_frames = 1
    else:
        prediction = 'No hands detected'
        no_hand_frames += 1
        gesture_history.clear()

        # Commit the last stable sign only after the hand is removed.
        if no_hand_frames >= NO_HAND_COMMIT_FRAMES and stable_word:
            sentence.append(stable_word)
            print(f'Word added: {stable_word}')

            stable_word = None
            stable_word_frames = 0
            prediction_history.clear()

    # Display
    display = frame.copy()

    # Top bar - current prediction
    cv2.rectangle(display, (0, 0), (640, 70), (0, 0, 0), -1)
    cv2.putText(display, prediction, (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3)

    # Bottom bar - sentence
    cv2.rectangle(display, (0, display.shape[0]-60),
                 (display.shape[1], display.shape[0]), (0, 0, 0), -1)
    sentence_text = sentence_to_text(sentence)
    cv2.putText(display, sentence_text, (10, display.shape[0]-20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

    cv2.imshow('MediBridge - Medical Sign Language', display)

    key = cv2.waitKey(25)
    if key == ord('q'):
        break
    if key == ord('c'):
        sentence = []
        prediction_history.clear()
        gesture_history.clear()
        stable_word = None
        stable_word_frames = 0
        no_hand_frames = 0
        print('Sentence cleared!')

cap.release()
cv2.destroyAllWindows()

final_sentence = sentence_to_text(sentence)
save_sentence(final_sentence)
print(f'Final sentence: {final_sentence}')