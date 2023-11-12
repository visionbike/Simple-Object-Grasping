from typing import Union
import cv2
import numpy as np
import rospy
from robot_control.TM_Robot import TMRobot
from tm_msgs.msg import *
from tm_msgs.srv import *
from camera import Camera


# define working space world Z
# MANUALLY EDIT
# [X, Y, Z, RX, RY, RZ]
Z_WORKING_SPACE = 128

# define the offset world Z
# MANUALLY EDIT
Z_OFFSET = 3

# define initial world point
# MANUALLY EDIT
WORLD_POINT_INIT = [-35.6, -421.65, 244.2, 180, 0, 90]

# define the end world point
WORLD_POINT_END = [-35.6, -421.65, 128, 180, 0, 90]

# define valid contour parameter limits in pixels
# MANUALLY EDIT
MIN_AREA = 200  # 10 x 10
MAX_AREA = 10000  # 100 x 100

# define thresholds for Otsu method
# MANUALLY EDIT
OTSU_SENSITIVITY = 22
OTSU_LOW_THRESH = 45
OTSU_HIGH_THRESH = 255

# define camera id
CAMERA_ID = 0

# define the global video frame
frame = None

# define the global 4 corners of ROI (top_left, top_right, bottom_left)
ROI_CORNERS = []

# define the calibration info path
PATH_CALIB_INFO = 'calibration_info'


def rospy_init_node(name):
    rospy.init_node(name)


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
        rospy.sleep(10)  # unit: ms
        # open the gripper
        tm_robot.set_IO('endeffector', 0, state='LOW')
        rospy.sleep(10)  # unit: ms
        # move to the intermediate position
        tm_robot.move(world_point_inter, move_type='PTP_T', speed=2.5, blend_mode=False)
        rospy.sleep(10)  # unit: ms
        # move to the end position
        tm_robot.move(world_point_end, move_type='PTP_T', speed=2.5, blend_mode=False)
        rospy.sleep(10)  # unit: ms
        # close the gripper
        tm_robot.set_IO('endeffector', 0, state='HIGH')
        rospy.sleep(10)  # unit: ms
        # move back to the initial position
        tm_robot.move(world_point_init, move_type='PTP_T', speed=2.5, blend_mode=False)


def click_event(event, x, y, flags, params):
    """
    The function for OpenCV click event.

    :param event: OpenCV click event.
    :param x: the clicked x value.
    :param y: the clicked y value.
    :param flags: the flags.
    :param params: the other parameters.
    """

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

    return ((pt2[0] - pt1[0]) ** 2 + (pt2[1] - pt1[1]) ** 2) ** 0.5


def compute_image_difference(bg, fg, min_thresh, max_thresh, sensitivity):
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
    # if ret is small, it means the there small different between bg and fg images
    if ret_ < sensitivity:
        # discard image
        # make the difference zero by subtracting backgrounds
        diff = cv2.absdiff(bg_gray, bg_gray)
    else:
        diff = cv2.GaussianBlur(otsu_thresh, (5, 5), 0)
    return diff


def get_valid_contours(contours, min_area, max_area):
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


def detect_contours(bg, fg, min_thresh, max_thresh, sensitivity, min_area, max_area):
    """
    Get the valid contours.

    :param bg: the input background image.
    :param fg: the foreground image.
    :param min_thresh: the minimum Otsu's threshold.
    :param max_thresh: the maximum Otsu's threshold.
    :param sensitivity: the Otsu's different sensitivity, if the value smaller this value,
            there a small difference in the image.
    :param min_area: the minimum area threshold.
    :param max_area: the maximum area threshold.
    :return: (contours, centroids)
    """

    # compute image difference
    diff = compute_image_difference(bg, fg, min_thresh, max_thresh, sensitivity)

    # find the contours
    # use RETR_EXTERNAL for only outer contours
    # use RETR_TREE for all the hierarchy
    contours, hierarchy = cv2.findContours(diff, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # identify the valid contours
    contour_ids = get_valid_contours(contours, min_area, max_area)

    # detect centroids, including
    # cx, cy: centroid point
    # w, h: width, height of min area rectangle bounding box
    # angle: the angle to detect the direction
    centroids = []
    for i, idx in enumerate(contour_ids):
        contour = contours[idx]

        # get centroid and angle
        centroid, size, angle = cv2.minAreaRect(contour)

        # [x, y, w, h, cx, cy, angle]
        centroids.append([centroid, size, angle])
    return contours, centroids


if __name__ == '__main__':
    # initiate ROS
    rospy_init_node('TM_CONTROL')
    # initiate TMRobot object
    tm_robot = TMRobot()

    # define window and click event
    window_name = 'Stream'
    cv2.namedWindow(window_name, cv2.WINDOW_FULLSCREEN)
    cv2.setMouseCallback(window_name, click_event)

    # load the calibration information
    camera = Camera(PATH_CALIB_INFO, 640, 480)

    # open webcam
    cap = cv2.VideoCapture(CAMERA_ID)

    # streaming
    frame_viz = None
    roi_bg = None
    rw, rh = 0, 0
    centroids = []
    ret = False
    while True:
        if ret:
            frame_viz = cv2.putText(frame_viz, '1. Select 3 corners of ROI', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36, 12, 255), 2)
            frame_viz = cv2.putText(frame_viz, '2. Put object inside the ROI', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36, 12, 255), 2)
            frame_viz = cv2.putText(frame_viz, '3. Press D to detection object', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36, 12, 255), 2)
            frame_viz = cv2.putText(frame_viz, '3. Press X to grasp object', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36, 12, 255), 2)
            frame_viz = cv2.putText(frame_viz, 'Press Q to exit', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36, 12, 255), 2)

            if len(ROI_CORNERS) == 3 and roi_bg is None:
                # align the ROI to rectangle ROI
                ROI_CORNERS[1][1] = ROI_CORNERS[0][1]
                ROI_CORNERS[2][0] = ROI_CORNERS[0][0]
                # compute ROI's width and height
                rw = compute_distance(ROI_CORNERS[1], ROI_CORNERS[0])
                rh = compute_distance(ROI_CORNERS[3], ROI_CORNERS[0])
                # get ROI's background
                roi_bg = frame[ROI_CORNERS[0][1]: ROI_CORNERS[0][1] + rh, ROI_CORNERS[0][0]: ROI_CORNERS[0][0] + rw, :]
            if roi_bg is not None:
                # draw ROI
                frame_viz = cv2.rectangle(frame_viz,
                                          ROI_CORNERS[0], (ROI_CORNERS[0][0] + rw, ROI_CORNERS[0][1] + rh),
                                          (36, 12, 255), 2)
            cv2.imshow(window_name, frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('d'):
                print('>>>> Detect objects...')
                # get the foreground image of ROI, make sure object is put inside ROI
                roi_fg = frame[ROI_CORNERS[0][1]: ROI_CORNERS[0][1] + rh, ROI_CORNERS[0][0]: ROI_CORNERS[0][0] + rw, :]
                # detect valid contours and their centroids
                contours, centroids = detect_contours(roi_bg, roi_fg, OTSU_LOW_THRESH, OTSU_HIGH_THRESH, OTSU_SENSITIVITY, MIN_AREA, MAX_AREA)
                if len(centroids) > 0:
                    # visualize the contours
                    for i, c in enumerate(centroids):
                        points = box = np.intp(cv2.boxPoints(c))
                        frame_viz = cv2.drawContours(frame_viz, [points], 0, (255, 0, 0), 1)
                        frame_viz = cv2.circle(frame_viz, (c[4], c[5]), 1, (255, 0, 0), 1)
                else:
                    print('No object found!')
            elif key == ord('x'):
                if len(centroids) > 0:
                    for i, c in enumerate(centroids):
                        try:
                            print('>>>> Robot Arm Control...')
                            XYZ_ = camera.compute_XYZ(c[0], c[1])
                            XYZ_ = np.squeeze(XYZ_, axis=1).tolist()
                            print(f'XYZ_: {XYZ_}')

                            WORLD_POINT_END = XYZ_ + WORLD_POINT_INIT[-3:]
                            # set Z for grasping
                            WORLD_POINT_END[2] = Z_WORKING_SPACE + Z_OFFSET
                            # set RZ for end-effector rotation
                            # MANUALLY EDIT
                            if c[2] < c[3]:
                                angle = 90 - c[-1]
                            else:
                                angle = -c[-1]
                            WORLD_POINT_END[-1] = angle
                            print(f'WORLD_POINT_END: {WORLD_POINT_END}')

                            WORLD_POINT_INTER = WORLD_POINT_END.copy()
                            WORLD_POINT_INTER[2] = WORLD_POINT_INIT[2]
                            print(f'WORLD_POINT_INTER: {WORLD_POINT_INTER}')

                            control_TM_arm(world_point_init=WORLD_POINT_INIT, world_point_inter=WORLD_POINT_INTER, world_point_end=WORLD_POINT_END)
                        except rospy.ROSInteruptException:
                            break
            elif key == ord('q'):
                print('>>>> Exit')
                break

        ret, frame = cap.read()
        frame_viz = frame.copy()
