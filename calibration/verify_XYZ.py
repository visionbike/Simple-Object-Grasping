import cv2
import numpy as np
import rospy
from TM_Robot import TM_Robot
from tm_msgs.msg import *
from tm_msgs.srv import *
from camera import Camera

TM = TM_Robot([])

rospy_init_node('TM_CONTROL')


class ImagePoint:
    x, y = 0, 0


def rospy_init_node(name):
    rospy.init_node(name)


def TM_control_arm(world_point_init, world_point_end):
    if not rospy.is_shutdown():
        # move to initial position
        TM.move_TM(world_point_init, Move_type='PTP_T', Speed=3.0, blend_Mode=False)
        rospy.sleep(3)
        # move to the end position
        TM.move_TM(world_point_end, Move_type='LINE_T', Speed=3.0, blend_Mode=False)
        rospy.sleep(3)
        # move back to the initial position
        TM.move_TM(world_point_init, Move_type='PTP_T', Speed=3.0, blend_Mode=False)


def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        param.x = x
        param.y = y


# define global touching image point
IMAGE_POINT = ImagePoint()

# define initial world point
# MANUALLY EDIT
WORLD_POINT_INIT = [-35.6, -421.65, 244.2, 180, 0, 90]

# define the end world point
WORLD_POINT_END = [-35.6, -421.65, 128, 180, 0, 90]

# define camera id
CAMERA_ID = 0

# define the calibration info path
PATH_CALIB_INFO = 'calibration_info'


if __name__ == '__main__':
    window_name = 'Verify XYZ'
    cv2.namedWindow(window_name, cv2.WINDOW_FULLSCREEN)
    cv2.setMouseCallback(window_name, click_event, IMAGE_POINT)

    camera = Camera(PATH_CALIB_INFO)

    cap = cv2.VideoCapture(CAMERA_ID)

    while True:
        ret, frame = cap.read()

        if ret:
            frame_viz = frame.copy()
            frame_viz = cv2.putText(frame_viz, '1. Left-click to select the point', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36, 12, 255), 2)
            frame_viz = cv2.putText(frame_viz, '2. Press X to continue processing', (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36, 12, 255), 2)
            frame_viz = cv2.putText(frame_viz, 'Press Q to exit', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36, 12, 255), 2)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('x'):
                try:
                    print('>>>> Robot Arm Control...')
                    XYZ_ = camera.calculate_XYZ(IMAGE_POINT.x, IMAGE_POINT.y)
                    print(f'Estimated XYZ: {XYZ_}')
                    WORLD_POINT_END = XYZ_ + WORLD_POINT_INIT[-3:]
                    TM_control_arm(world_point_init=WORLD_POINT_INIT, world_point_end=WORLD_POINT_END)
                except rospy.ROSInteruptException:
                    pass
            elif key == ord('q'):
                print('>>>> Exit')
                break
            cv2.imshow(window_name, frame_viz)
        else:
            break
