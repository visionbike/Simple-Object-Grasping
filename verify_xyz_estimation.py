import cv2
import numpy as np
import rospy
from robot_control.TM_Robot import TMRobot
from tm_msgs.msg import *
from tm_msgs.srv import *
from camera import Camera


# define working space world Z
# MANUALLY EDIT
Z_WORKING_SPACE = 128

# define the offset world Z
# MANUALLY EDIT
Z_OFFSET = 3

# define initial world point
# MANUALLY EDIT
WORLD_POINT_INIT = [-35.6, -421.65, 244.2, 180, 0, 90]

# define the end world point
WORLD_POINT_END = [-35.6, -421.65, 128, 180, 0, 90]

# define global touching image point
PX, PY = 0, 0

# define camera id
CAMERA_ID = 0

# define the global video frame
frame = None

# define the 'clicked' flag
clicked = False

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
        rospy.sleep(10)     # unit: ms
        # move to the intermediate position
        tm_robot.move(world_point_inter, move_type='PTP_T', speed=2.5, blend_mode=False)
        rospy.sleep(10)     # unit: ms
        # move to the end position
        tm_robot.move(world_point_end, move_type='PTP_T', speed=2.5, blend_mode=False)
        rospy.sleep(25)     # unit: ms
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

    global frame, clicked, PX, PY
    if event == cv2.EVENT_LBUTTONDOWN:
        PX = x
        PY = y
        clicked = True
        

if __name__ == '__main__':
    # initiate ROS
    rospy_init_node('TM_CONTROL')
    # initiate TMRobot object
    tm_robot = TMRobot()

    # define window and click event
    window_name = 'Verify XYZ'
    cv2.namedWindow(window_name, cv2.WINDOW_FULLSCREEN)
    cv2.setMouseCallback(window_name, click_event)

    camera = Camera(PATH_CALIB_INFO, 640, 480)

    cap = cv2.VideoCapture(CAMERA_ID)
    ret, frame = cap.read()
    
    while True:
        if ret:
            frame = cv2.putText(frame, '1. Left-click to select the point', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36, 12, 255), 2)
            frame = cv2.putText(frame, '2. Press X to continue processing', (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36, 12, 255), 2)
            frame = cv2.putText(frame, 'Press Q to exit', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36, 12, 255), 2)
            if clicked:
                frame = cv2.circle(frame, (PX, PY), 2, (0, 0, 255), 2)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('x'):
                try:
                    print('>>>> Robot Arm Control...')
                    XYZ_ = camera.compute_XYZ(PX, PY)
                    XYZ_ = np.squeeze(XYZ_, axis=1).tolist()
                    print(f'XYZ_: {XYZ_}')

                    WORLD_POINT_END = XYZ_ + WORLD_POINT_INIT[-3:]
                    WORLD_POINT_END[2] = Z_WORKING_SPACE + Z_OFFSET
                    print(f'WORLD_POINT_END: {WORLD_POINT_END}')

                    WORLD_POINT_INTER = WORLD_POINT_END.copy()
                    WORLD_POINT_INTER[2] = WORLD_POINT_INIT[2]
                    print(f'WORLD_POINT_INTER: {WORLD_POINT_INTER}')

                    control_TM_arm(world_point_init=WORLD_POINT_INIT, world_point_inter=WORLD_POINT_INTER, world_point_end=WORLD_POINT_END)
                except rospy.ROSInteruptException:
                    pass
            elif key == ord('q'):
                print('>>>> Exit')
                break
            # frame_viz = frame.copy()
            cv2.imshow(window_name, frame)
        else:
            break
        
        ret, frame = cap.read()
