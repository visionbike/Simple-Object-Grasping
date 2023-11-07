from pathlib import Path
import numpy as np
import cv2
from simple_pattern_detector import *

__all__ = ['PointDetector']


class PointDetector:

    def __init__(self, camera_id, path_env_info):
        self.camera_id = camera_id
        self.path_env_info = Path(path_env_info)
        self.path_env_info.mkdir(parents=True, exist_ok=True)

    def capture_initial_image(self, mode='bg'):
        print(f">>>> Capturing {'background' if mode == 'bg' else 'Foreground'} Image...")
        if mode == 'bg':
            window_name = 'Initial Background'
            fname = 'bg.jpg'
        else:
            window_name = 'Initial Foreground'
            fname = 'fg.jpg'
        cv2.namedWindow(window_name, cv2.WINDOW_FULLSCREEN)

        # define video capture
        cap = cv2.VideoCapture(self.camera_id)
        while True:
            # capture video frame by frame
            ret, frame = cap.read()
            if ret:
                frame_viz = frame.copy()
                frame_viz = cv2.putText(frame_viz, "Press X to capture image", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36, 12, 255), 2)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('x'):
                    cv2.imwrite(str(self.path_env_info / fname), frame)
                    print('Saved frame!')
                    cap.release()
                    cv2.destroyAllWindows()
                    break
                cv2.imshow(window_name, frame_viz)
            else:
                break

    def detect_patterns(self):
        window_name = 'Points'
        cv2.namedWindow(window_name, cv2.WINDOW_FULLSCREEN)

        print('>>> Detecting Patterns...')
        fg = cv2.imread(str(self.path_env_info / 'fg.jpg'))
        bg = cv2.imread(str(self.path_env_info / 'bg.jpg'))
        pr = SimplePatternDetector()
        pattern_count, points_detected, im_out = pr.run_detection(fg, bg)
        if pattern_count == 0:
            print(f'No pattern detected')
            return False
        print(f'{pattern_count} patterns detected')

        print('>>> Detect Image Center...')
        h, w = bg.shape[: 2]
        im_center = [int(w // 2), int(h // 2)]
        # draw image center
        im_out = cv2.circle(im_out, im_center, 2, (0, 0, 255), 2)
        # draw the text
        im_out = cv2.putText(im_out, 'C', (im_center[0], im_center[1] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # visualize
        cv2.imshow(window_name, im_out)
        cv2.waitKey(5000)
        cv2.destroyAllWindows()

        return points_detected, im_center

    def stream_patterns(self, points_detected, im_center):
        print('>>>> Streaming Patterns...')
        window_name = 'Streaming'
        cv2.namedWindow(window_name, cv2.WINDOW_FULLSCREEN)

        # define video capture
        cap = cv2.VideoCapture(self.camera_id)
        while True:
            # capture video frame by frame
            ret, frame = cap.read()
            if ret:
                frame_viz = frame.copy()
                # draw center of patterns
                for i, pt in enumerate(points_detected):
                    frame_viz = cv2.circle(frame_viz, (pt[-2], pt[-1]), 2, (255, 0, 0), 2)
                    frame_viz = cv2.putText(frame_viz, f'{i}', (pt[-2], pt[-1] + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                # draw image center
                frame_viz = cv2.circle(frame_viz, (im_center[0], im_center[1]), 2, (0, 0, 255), 2)
                frame_viz = cv2.putText(frame_viz, 'C', (im_center[0], im_center[1] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                frame_viz = cv2.putText(frame_viz, 'Press Q to close', (0, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print('Exit!')
                    break
                cv2.imshow(window_name, frame_viz)
            else:
                break

        cap.release()
        cv2.destroyAllWindows()

    def save_image_points(self, points_detected, im_center):
        print('>>>> Saving Image Points...')
        points = [im_center]
        for pt in points_detected:
            points.append(pt[-2:])
        points = np.array(points, dtype=np.float32)
        print(f'Image points:\n{points}')
        np.save(str(self.path_env_info / 'image_coord.npy'), points)
