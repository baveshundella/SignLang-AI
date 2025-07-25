import cv2
import os

DATASET_PATH = '../dataset'
LABELS = [chr(i) for i in range(65, 91)]  # A-Z

def create_label_dirs():
    for label in LABELS:
        os.makedirs(os.path.join(DATASET_PATH, label), exist_ok=True)

def main():
    create_label_dirs()
    cap = cv2.VideoCapture(0)
    current_label = 'A'
    count = 0

    print("Press the letter key (A-Z) to save an image for that label.")
    print("Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        # Draw ROI rectangle
        x1, y1, x2, y2 = 100, 100, 324, 324
        roi = frame[y1:y2, x1:x2]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(frame, f'Label: {current_label} | Count: {count}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2, cv2.LINE_AA)
        cv2.imshow("Frame", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif 65 <= key <= 90 or 97 <= key <= 122:  # A-Z or a-z
            label = chr(key).upper()
            if label in LABELS:
                current_label = label
                img_path = os.path.join(DATASET_PATH, label, f"{label}_{count}.jpg")
                cv2.imwrite(img_path, roi)
                print(f"Saved {img_path}")
                count += 1

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
