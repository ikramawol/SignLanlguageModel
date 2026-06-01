import os
import cv2

DATA_DIR = './video_data'
os.makedirs(DATA_DIR, exist_ok=True)

CLASSES = {
    0: 'doctor',
    1: 'nurse',
    2: 'pain',
    3: 'medicine',
    
}

VIDEOS_PER_CLASS = 5      # 5 videos per word
FRAMES_PER_VIDEO = 60     # 2 seconds per video at 30fps

cap = cv2.VideoCapture(0)

for class_id, class_name in CLASSES.items():
    class_dir = os.path.join(DATA_DIR, class_name)
    os.makedirs(class_dir, exist_ok=True)

    print(f'\n>>> Word: {class_name.upper()} ({class_id+1}/10)')

    for vid_num in range(VIDEOS_PER_CLASS):
        # Wait for user to get ready
        while True:
            ret, frame = cap.read()
            cv2.putText(frame, f'Word: {class_name.upper()}', (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
            cv2.putText(frame, f'Video {vid_num+1}/{VIDEOS_PER_CLASS}', (10, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)
            cv2.putText(frame, 'Press Q when ready to record', (10, 130),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)
            cv2.imshow('MediBridge - Data Collection', frame)
            if cv2.waitKey(25) == ord('q'):
                break

        # Countdown 3..2..1
        for count in [3, 2, 1]:
            for _ in range(20):
                ret, frame = cap.read()
                cv2.putText(frame, f'Starting in {count}...', (10, 200),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)
                cv2.imshow('MediBridge - Data Collection', frame)
                cv2.waitKey(25)

        # Record video
        frames = []
        for i in range(FRAMES_PER_VIDEO):
            ret, frame = cap.read()
            frames.append(frame.copy())

            # Show recording indicator
            display = frame.copy()
            cv2.putText(display, f'RECORDING... {class_name.upper()}', (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            progress = int((i / FRAMES_PER_VIDEO) * display.shape[1])
            cv2.rectangle(display, (0, display.shape[0]-20),
                         (progress, display.shape[0]), (0, 0, 255), -1)
            cv2.imshow('MediBridge - Data Collection', display)
            cv2.waitKey(25)

        # Save video
        save_path = os.path.join(class_dir, f'{vid_num}.mp4')
        h, w = frames[0].shape[:2]
        out = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (w, h))
        for f in frames:
            out.write(f)
        out.release()
        print(f'  Saved: {save_path}')

    print(f'Done with {class_name}!')

cap.release()
cv2.destroyAllWindows()
print('\nAll done! Videos saved in video_data/')