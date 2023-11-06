from pathlib import Path
import cv2

CALIB_IMG_DIR = Path('calibration_images')
CALIB_IMG_DIR.mkdir(parents=True, exist_ok=True)

# define visualization window
window_name = 'Capture'
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

CAMERA_ID = 0

if __name__ == '__main__':
    # define video capture
    cap = cv2.VideoCapture(CAMERA_ID)
    i = 0
    while True:
        # capture video frame by frame
        ret, frame = cap.read()
        if ret:
            frame = cv2.putText(frame, "Press X to capture image", (15, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36, 255, 12), 1)
            frame = cv2.putText(frame, "Press Q to exit", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36, 255, 12), 1)
            frame = cv2.putText(frame, f"Saved frames: {i}", (15, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36, 255, 12), 1)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('x'):
                i += 1
                cv2.imwrite(str(CALIB_IMG_DIR / f'cap_{i}.jpg'), frame)
                print('>>>> Saved frame!')
            elif key == ord('q'):
                print('>>> Exit')
                break
            cv2.imshow(window_name, frame)
        else:
            break

    cap.release()
    cv2.destroyAllWindows()
