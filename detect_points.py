from pathlib import Path
import numpy as np
import cv2
from detector import ShapeDetector


# define environment info path
# MANUALLY EDIT
PATH_ENV_INFO = 'environment_info'

# define camera id
# MANUALLY EDIT
CAMERA_ID = 0

# define valid contour parameter limits in pixels
# MANUALLY EDIT
MIN_AREA = 200  # 10 x 10
MAX_AREA = 10000  # 100 x 100

# define aspect ratio width/height
# MANUALLY EDIT
MIN_ASPECT_RATIO = 0.25  # 1/5
MAX_ASPECT_RATIO = 5.0

# define thresholds for Otsu method
# MANUALLY EDIT
OTSU_SENSITIVITY = 22
OTSU_LOW_THRESH = 45
OTSU_HIGH_THRESH = 255


class PointDetector:

    def __init__(self, camera_id: int, path_env_info: str):
        """

        :param camera_id: the camera id.
        :param path_env_info: the path to store the image points.
        """

        self.path_env_info = Path(path_env_info)
        self.path_env_info.mkdir(parents=True, exist_ok=True)
        self.camera_id = camera_id

        self.center = []
        self.centroids = []
        self.im_viz = None

    def _draw_points(self, im: np.array):
        """
        Draw the points to the input image.

        :param im: the input image.
        :return: the drawn image.
        """

        # draw image center
        im = cv2.circle(im, (self.center[0], self.center[1]), 2, (0, 0, 255), 2)
        im = cv2.putText(im, 'C', (self.center[0], self.center[1] + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        # draw the centroids
        for i, c in enumerate(self.centroids):
            im = cv2.rectangle(im, (c[0], c[1]), (c[0] + c[2], c[1] + c[3]), (0, 255, 0), 2)
            im = cv2.circle(im, (c[-2], c[-1]), 2, (0, 255, 0), 2)
            im = cv2.putText(im, f'{i}', (c[-2], c[-1] + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        return im

    def capture_initial_image(self, mode='bg'):
        """
        Capture the background and foreground image.

        :param mode: 'bg' to detect background and 'fg' to detect foreground. Default: 'bg'.
        """
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

    def detect_points(self):
        """
        Detection the centroids of detected patterns and the image center.
        """

        print('>>> Detecting Patterns...')
        fg = cv2.imread(str(self.path_env_info / 'fg.jpg'))
        bg = cv2.imread(str(self.path_env_info / 'bg.jpg'))
        pr = ShapeDetector()
        contours, self.centroids = pr.detect_shape(fg, bg,
                                              OTSU_SENSITIVITY, OTSU_HIGH_THRESH, OTSU_SENSITIVITY,
                                              MIN_ASPECT_RATIO, MAX_ASPECT_RATIO,
                                              MIN_AREA, MAX_AREA)
        print(f'{len(self.centroids)} patterns detected.')
        if len(self.centroids) == 0:
            return False

        print('>>> Detect Image Center...')
        h, w = bg.shape[: 2]
        self.center = [int(w // 2), int(h // 2)]

        # visualization
        fg_viz = fg.copy()
        fg_viz = self._draw_points(fg_viz)

        window_name = 'Points'
        cv2.namedWindow(window_name, cv2.WINDOW_FULLSCREEN)
        cv2.imshow(window_name, fg_viz)
        cv2.waitKey(5000)
        cv2.destroyAllWindows()

        self.im_viz = fg_viz

    def stream_points(self):
        """
        Streaming webcam with drawn detected points.
        """

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
                frame_viz = cv2.putText(frame_viz, 'Press Q to close', (0, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                # draw center of patterns
                frame_viz = self._draw_points(frame_viz)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print('Exit!')
                    break
                cv2.imshow(window_name, frame_viz)
            else:
                break

        cap.release()
        cv2.destroyAllWindows()

    def save_image_points(self):
        """
        Save the detected image points.
        """
        print('>>>> Saving Image Points...')
        points = [self.center]
        for pt in self.centroids:
            points.append(pt[-2:])
        points = np.array(points, dtype=np.float32)
        print(f'Image points:\n{points}')
        np.save(str(self.path_env_info / 'image_coord.npy'), points)
        if self.im_viz:
            cv2.imwrite(str(self.path_env_info / 'image_coord.png'), self.im_viz)


if __name__ == '__main__':
    pd = PointDetector(CAMERA_ID, PATH_ENV_INFO)

    pd.capture_initial_image(mode='bg')
    pd.capture_initial_image(mode='fg')

    pd.detect_points()
    pd.save_image_points()

    # streaming to get real-world coordinate using TM Robot, value in mm
    pd.stream_points()
