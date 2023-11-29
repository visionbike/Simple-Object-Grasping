from typing import Union
import time
import operator
import cv2
import numpy as np
import rospy
from robot_control.TM_Robot import TMRobot
from tm_msgs.msg import *
from tm_msgs.srv import *
from camera import Camera
from detector import YoloObjectDetector

# define the YOLO model path
# MANUALLY EDIT
MODEL_PATH = './models'

# define working space world Z
# MANUALLY EDIT
Z_WORKING_SPACE = 122

# define the offset world Z
# MANUALLY EDIT
Z_OFFSET = -20

# define initial world point
# [X, Y, Z, RX, RY, RZ]
# MANUALLY EDIT
WORLD_POINT_INIT = [530.81, 102.63, 171.43, -180, 0, 90]

# define the end world point
WORLD_POINT_END = [-35.6, -421.65, 128, 180, 0, 90]

# MANUALLY EDIT
THRESH_BOX_CONF_MIN = 0.25
THRESH_BOX_CONF_MAX = 0.5
THRESH_CLASS_CONF = 0.5

# define valid contour parameter limits in pixels
# MANUALLY EDIT
MIN_AREA = 180  # 10 x 10
MAX_AREA = 100000  # 100 x 100

# define thresholds for Otsu method
# MANUALLY EDIT
OTSU_SENSITIVITY = 20
OTSU_LOW_THRESH = 40
OTSU_HIGH_THRESH = 255

# define the range in pixel to detect object
# MANUALLY EDIT
THRESH_PAD_X = 30

# define camera id
CAMERA_ID = 0

# define the global video frame
frame = None

# define the global 4 corners of ROI (top_left, top_right, bottom_left)
ROI_CORNERS = []

# define the calibration info path
PATH_CALIB_INFO = 'calibration_info'


def rospy_init_node(name: str):
    """
    Initiate ROS node

    :param name: node name
    :return:
    """
    rospy.init_node(name)


def control_TM_conveyor(tm_robot: TMRobot, pin: int, state: str = 'HIGH'):
    """
    Control the TM conveyor.

    :param tm_robot: the TM robot object.
    :param pin: the PIN that connect the conveyor.
    :param state: 'HIGH' to start and 'LOW' to stop the conveyor. Default: 'HIGH'.
    :return:
    """

    if not rospy.is_shutdown():
        tm_robot.set_IO('controlbox', pin, state)
        rospy.sleep(3)
    return True if state == 'HIGH' else False


def control_TM_arm(tm_robot: TMRobot, world_point_init: list, world_point_inter: list, world_point_end: list):
    """
    Control the TM Arm

    :param tm_robot: the TM robot object.
    :param world_point_init: the initial world point.
    :param world_point_inter: the intermediate world point.
    :param world_point_end: the end world point.
    """
    if not rospy.is_shutdown():
        # move to initial position
        tm_robot.move(world_point_init, move_type='PTP_T', speed=2.5, blend_mode=False)
        rospy.sleep(3)  # unit: s
        # open the gripper
        tm_robot.set_IO('endeffector', 0, state='LOW')
        rospy.sleep(3)  # unit: s
        # move to the intermediate position
        tm_robot.move(world_point_inter, move_type='PTP_T', speed=2.5, blend_mode=False)
        rospy.sleep(3)  # unit: s
        # move to the end position
        tm_robot.move(world_point_end, move_type='PTP_T', speed=2.5, blend_mode=False)
        rospy.sleep(3)  # unit: s
        # close the gripper
        tm_robot.set_IO('endeffector', 0, state='HIGH')
        rospy.sleep(10)  # unit: s
        # move back to the intermediate position
        tm_robot.move(world_point_inter, move_type='PTP_T', speed=2.5, blend_mode=False)
        rospy.sleep(3)  # unit: s
        # move back to the initial position
        tm_robot.move(world_point_init, move_type='PTP_T', speed=2.5, blend_mode=False)
        rospy.sleep(3)  # unit: s
        # close the gripper
        tm_robot.set_IO('endeffector', 0, state='LOW')


def click_event(event, x, y, flags, params):
    """
    The function for OpenCV click event.

    :param event: OpenCV click event.
    :param x: the clicked x value.
    :param y: the clicked y value.
    :param flags: the flags.
    :param params: the other parameters.
    """
    global ROI_CORNERS
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(ROI_CORNERS) < 3:
            ROI_CORNERS.append([int(x), int(y)])


def compute_distance(pt1: Union[list, np.array], pt2: Union[list, np.array]):
    """
    Compute distance between two points

    :param pt1: the first point (x1, y1).
    :param pt2: the second point (x2, y2).
    :return: the length of two points.
    """

    return int(((pt2[0] - pt1[0]) ** 2 + (pt2[1] - pt1[1]) ** 2) ** 0.5)


def compute_image_difference(bg: np.array, fg: np.array, min_thresh: int, max_thresh: int, sensitivity: int):
    """
    Compute the difference between background and foreground image using Otsu threshold method.

    :param bg: the input background image.
    :param fg: the input foreground image.
    :param min_thresh: the minimum Otsu's threshold.
    :param max_thresh: the maximum Otsu's threshold.
    :param sensitivity: the Otsu's different sensitivity, if the value smaller this value,
            there a small difference in the image.
    :return: the difference image.
    """

    # convert to grayscale
    bg_gray = cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY)
    fg_gray = cv2.cvtColor(fg, cv2.COLOR_BGR2GRAY)
    
    # compute the difference of images
    diff_gray = cv2.absdiff(bg_gray, fg_gray) 

    # blur the result to remove noise
    diff_gray_blur = cv2.GaussianBlur(diff_gray, (5, 5), 0)

    # find Otsu's threshold image
    ret_, otsu_thresh = cv2.threshold(diff_gray_blur, min_thresh, max_thresh, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # ret_ is considered as the threshold value
    # if ret_ is small, it means the there small different between bg and fg images
    if ret_ < sensitivity:
        # discard image
        # make the difference zero by subtracting backgrounds
        diff = cv2.absdiff(bg_gray, bg_gray)
    else:
        diff = cv2.GaussianBlur(otsu_thresh, (5, 5), 0)
    return diff


def get_valid_contours(contours: list, min_area: int, max_area: int):
    """
    Get the valid contours in a range of area value.
    :param contours: the input contours.
    :param min_area: the minimum area threshold.
    :param max_area: the maximum area threshold.
    :return: the valid indices.
    """
    valid_ids = []
    for i, contour in enumerate(contours):
        contour_area = cv2.contourArea(contour)

        # discard contours that out of range in terms of contour area
        if min_area < contour_area < max_area:
            valid_ids.append(i)
    return valid_ids


def detect_contours(bg: np.array, fg: np.array, class_ids: list, box_confs: list, boxes: list, min_thresh: int, max_thresh: int, sensitivity: int, min_area: int, max_area: int):
    """
    Detect YOLO detected objects' centroids and grasping orientation.

    :param bg: the input background.
    :param fg: the input foreground.
    :param class_ids: the class ids of detected objects.
    :param box_confs: the bounding confidence values of detected objects.
    :param boxes: the bounding boxes of detected objects.
    :param min_thresh: the minimum OTSU's threshold.
    :param max_thresh: the maximum OTSU's threshold.
    :param sensitivity: the OTSU's threshold for detect the difference between `bg` and `fg`.
    :param min_area: the minimum contour area threshold.
    :param max_area: the maximum contour area threshold.
    :return: (class_ids, box_confs, boxes, centroids)
    """

    valid_ids = []
    centroids = []
    for i, box in enumerate(boxes):
        # compute image difference
        diff = compute_image_difference(
            bg[box[1]: box[1] + box[3], box[0]: box[0] + box[2], :],
            fg[box[1]: box[1] + box[3], box[0]: box[0] + box[2], :],
            min_thresh, max_thresh, sensitivity)

        # find the contours
        # use RETR_EXTERNAL for only outer contours
        # use RETR_TREE for all the hierarchy
        contours, hierarchy = cv2.findContours(diff, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # identify the valid contours
        contour_ids = get_valid_contours(contours, min_area, max_area)

        if len(contour_ids) > 0:
            # detect centroids, including
            # (cx, cy): centroid point
            # (w, h): width, height of min area rectangle bounding box
            # angle: the angle to detect the direction
            contour = contours[0]

            # get center and angle of minimum area rectangle
            center, size, angle = cv2.minAreaRect(contour)

            # estimate centroid of contours
            M = cv2.moments(contour)
            cx = int(box[0] + M['m10'] / M['m00'])
            cy = int(box[1] + M['m01'] / M['m00'])

            centroids.append([[cx, cy], list(size), angle])
            valid_ids.append(i)
    if len(valid_ids) > 1:
        # list indices function
        class_ids = [class_ids[i] for i in valid_ids]
        box_confs = [box_confs[i] for i in valid_ids]
        boxes = [boxes[i] for i in valid_ids]
        return class_ids, box_confs, boxes, centroids
    return class_ids, box_confs, boxes, centroids


if __name__ == '__main__':
    # initiate ROS
    rospy_init_node('TM_CONTROL')
    # initiate TMRobot object
    tm_robot = TMRobot()

    # stop conveyor first
    control_TM_conveyor(tm_robot, 4, 'LOW')

    # initiate YOLO model
    model_yolo = YoloObjectDetector(MODEL_PATH, use_cuda=False)

    # define window and click event
    window_name = 'Stream'
    cv2.namedWindow(window_name, cv2.WINDOW_FULLSCREEN)
    cv2.setMouseCallback(window_name, click_event)

    # load the calibration information
    camera = Camera(PATH_CALIB_INFO, 640, 480)

    # open webcam
    cap = cv2.VideoCapture(CAMERA_ID)
    cap.set(cv2.CAP_PROP_FPS, 5)

    # streaming
    frame_viz = None
    roi_bg = None
    rw, rh = 0, 0
    centroids = []
    boxes = []
    class_ids = []
    ret, ret_conveyor = False, False
    while True:

        if ret:
            frame_viz = frame.copy()
            frame_viz = cv2.putText(frame_viz, '1. Select 3 corners of ROI', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36, 12, 255), 2)
            frame_viz = cv2.putText(frame_viz, '2. Put object outside the ROI', (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36, 12, 255), 2)
            frame_viz = cv2.putText(frame_viz, '3. Press X to start conveyor and detect', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36, 12, 255), 2)
            frame_viz = cv2.putText(frame_viz, 'Press Q to exit', (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36, 12, 255), 2)

            if 0 < len(ROI_CORNERS) < 3:
                # draw point
                for rc in ROI_CORNERS:
                    frame_viz = cv2.circle(frame_viz, rc, 2, (36, 12, 255), 2)
            elif len(ROI_CORNERS) == 3 and roi_bg is None:
                # align the ROI to rectangle ROI
                ROI_CORNERS[1][1] = ROI_CORNERS[0][1]
                ROI_CORNERS[2][0] = ROI_CORNERS[0][0]
                # compute ROI's width and height
                rw = compute_distance(ROI_CORNERS[1], ROI_CORNERS[0])
                rh = compute_distance(ROI_CORNERS[2], ROI_CORNERS[0])
                # get ROI's background
                roi_bg = frame[ROI_CORNERS[0][1]: ROI_CORNERS[0][1] + rh, ROI_CORNERS[0][0]: ROI_CORNERS[0][0] + rw, :]

            if roi_bg is not None:
                # draw ROI
                frame_viz = cv2.putText(frame_viz, 'ROI', (ROI_CORNERS[0][0], ROI_CORNERS[0][1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36, 12, 255), 2)
                frame_viz = cv2.rectangle(frame_viz, ROI_CORNERS[0], (ROI_CORNERS[0][0] + rw, ROI_CORNERS[0][1] + rh), (36, 12, 255), 2)
            
            if ret_conveyor and len(centroids) == 0:
                print('>>>> Detect objects...')
                # get the foreground image of ROI, make sure object is put inside ROI
                roi_fg = frame[ROI_CORNERS[0][1]: ROI_CORNERS[0][1] + rh, ROI_CORNERS[0][0]: ROI_CORNERS[0][0] + rw, :]
                # detect objects
                class_ids, box_confs, boxes = model_yolo.detect_objects(roi_fg, min_box_conf_thresh=THRESH_BOX_CONF_MIN, max_box_conf_thresh=THRESH_BOX_CONF_MAX, class_conf_thresh=THRESH_CLASS_CONF)
                
            if len(boxes) > 0: 
                # rw is the width of ROI
                # if the center point of the first detected object's bounding box reach over the rw / 2.5 line (near-center line)
                # then stop the conveyor
                if (boxes[0][0] + ROI_CORNERS[0][0] + boxes[0][2] // 2 >= ROI_CORNERS[0][0] + rw / 2.5):
                    # stop conveyor
                    ret_conveyor = control_TM_conveyor(tm_robot, 4, 'LOW')

                    # detect centroids                    
                    class_ids, box_confs, boxes, centroids = detect_contours(roi_bg, roi_fg, [class_ids[0]], [box_confs[0]], [boxes[0]], OTSU_LOW_THRESH, OTSU_HIGH_THRESH, OTSU_SENSITIVITY, MIN_AREA, MAX_AREA)

                    # remap the centroids from ROI to full image
                    for i in range(len(centroids)):
                        centroids[i][0][0] += ROI_CORNERS[0][0]
                        centroids[i][0][1] += ROI_CORNERS[0][1]

                # visualize the centroids
                for i, c in enumerate(centroids):
                    frame_viz = cv2.putText(frame_viz, f'{model_yolo.get_class_name(class_ids[i])}', (c[0][0], c[0][1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    # frame_viz = cv2.rectangle(frame_viz, box[:2], (box[0] + box[2], box[1] + box[3]), (255, 12, 36), 2)
                    points = np.intp(cv2.boxPoints(c))
                    frame_viz = cv2.drawContours(frame_viz, [points], 0, (255, 0, 0), 2)
                    frame_viz = cv2.circle(frame_viz, (c[0][0], c[0][1]), 1, (0, 255, 0), 2)

            cv2.imshow(window_name, frame_viz)

            key = cv2.waitKey(1) & 0xFF
                
            if len(ROI_CORNERS) == 3 and key == ord('x') and not ret_conveyor:
                print('>>>> Start conveyor...')
                ret_conveyor = control_TM_conveyor(tm_robot, 4, 'HIGH')
                first_time = True
            elif key == ord('q'):
                print('>>>> Exit')
                break

            if len(centroids) > 0:
                
                for i, c in enumerate(centroids):
                    print('>>>> Robot Arm Control...')
                    XYZ_ = camera.compute_XYZ(c[0][0], c[0][1])
                    XYZ_ = np.squeeze(XYZ_, axis=1).tolist()
                    print(f'XYZ_: {XYZ_}')

                    WORLD_POINT_END = XYZ_ + WORLD_POINT_INIT[-3:]
                    # set Z for grasping
                    WORLD_POINT_END[2] = Z_WORKING_SPACE + Z_OFFSET
                    # set RZ for end-effector rotation
                    print(c[2], c[1])
                    if c[1][0] < c[1][1]:
                        # if width < height
                        angle = 90 - c[2]
                    else:
                        angle = -c[2]
                    print(angle)
                    WORLD_POINT_END[-1] = WORLD_POINT_END[-1] + angle
                    print(f'WORLD_POINT_END: {WORLD_POINT_END}')

                    WORLD_POINT_INTER = WORLD_POINT_END.copy()
                    WORLD_POINT_INTER[2] = WORLD_POINT_INIT[2]
                    print(f'WORLD_POINT_INTER: {WORLD_POINT_INTER}')
                    try:
                        
                        control_TM_arm(tm_robot=tm_robot,
                                    world_point_init=WORLD_POINT_INIT,
                                    world_point_inter=WORLD_POINT_INTER,
                                    world_point_end=WORLD_POINT_END)
                    except rospy.ROSInteruptException:
                        break
                    
                # reset centroids list
                centroids = []
                boxes = []

                time.sleep(10)

                frame_viz, frame = None, None
                ret = False


        ret, frame = cap.read()

