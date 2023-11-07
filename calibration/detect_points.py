import numpy as np
import cv2
from point_detector import *

# define environment info path
# MANUALLY EDIT
PATH_ENV_INFO = 'environment_info'

# define camera id
# MANUALLY EDIT
CAMERA_ID = 0

if __name__ == '__main__':
    pd = PointDetector(CAMERA_ID, PATH_ENV_INFO)

    pd.capture_initial_image(mode='bg')
    pd.capture_initial_image(mode='fg')

    points_detected, im_center = pd.detect_patterns()
    pd.save_image_points(points_detected, im_center)

    # streaming to get real-world coordinate using TM Robot, value in mm
    pd.stream_patterns(points_detected, im_center)
