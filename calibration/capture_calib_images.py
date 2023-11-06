from pathlib import Path
import cv2

CALIB_IMG_DIR = Path('calibration_images')
CALIB_IMG_DIR.mkdir(parents=True, exist_ok=True)

# define visualization window
window_name = 'Capture'
cv2.namedWindow(window_name, cv2.WINDOW_FULLSCREEN)


CAMERA_ID = 0

if __name__ == '__main__':
    # define video capture
    cap = cv2.VideoCapture(CAMERA_ID)
    i = 0
    while True:
        # capture video frame by frame
        ret, frame = cap.read()
        if ret:
            frame_viz = frame.copy()
            frame_viz = cv2.putText(frame_viz, "Press X to capture image", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36, 12, 255), 2)
            frame_viz = cv2.putText(frame_viz, "Press Q to exit", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36, 12, 255), 2)
            frame_viz = cv2.putText(frame_viz, f"Saved frames: {i}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36, 12, 255), 2)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('x'):
                i += 1
                cv2.imwrite(str(CALIB_IMG_DIR / f'cap_{i}.jpg'), frame)
                print('>>>> Saved frame!')
            elif key == ord('q'):
                print('>>> Exit')
                break
            cv2.imshow(window_name, frame_viz)
        else:
            break

    cap.release()
    cv2.destroyAllWindows()
