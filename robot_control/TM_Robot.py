from typing import Optional
import rospy
from tm_msgs.msg import *
from tm_msgs.srv import *
import math

__all__ = ['TMRobot']


class TMRobot:
    """
    TM Robot control via ROS's TM Driver
    """

    rospy.wait_for_service('tm_driver/set_positions')
    rospy.wait_for_service('tm_driver/set_io')
    set_positions = rospy.ServiceProxy('tm_driver/set_positions', SetPositions)
    set_io = rospy.ServiceProxy('tm_driver/set_io', SetIO)

    def __init__(self, pose: Optional[list] = None):
        self.pose_init = pose

    def move(self, pose: list, move_type: str = 'PTP_J', speed: float = 1.0, blend_mode: bool = True):
        """
        Move the robot given the end point's pose.

        :param pose: the end point's pose, include [X, Y, Z, Rx, Ry, Rz].
        :param move_type: point-to-point in Joints (PTP_J), point-to-point in translation (PTP_T) or line in translation (LINE_T) mode. Default: 'PTP_J'.
        :param speed: movement speed.
        :param blend_mode: whether to apply blending or not. Default: True.
        """
        blend_percent = 0
        print(">>> Moving to next point...")

        if blend_mode:
            print('>>>> Blending mode: ON')
            blend_percent = 100

        print("Moving type: " + move_type)

        if move_type == 'PTP_J':
            J1 = (pose[0] / 180.0) * math.pi            # unit: radian
            J2 = (pose[1] / 180.0) * math.pi
            J3 = (pose[2] / 180.0) * math.pi
            J4 = (pose[3] / 180.0) * math.pi
            J5 = (pose[4] / 180.0) * math.pi
            J6 = (pose[5] / 180.0) * math.pi
            point = [J1, J2, J3, J4, J5, J6]
        else:
            S_x = pose[0] / 1000.0                      # unit: mm
            S_y = pose[1] / 1000.0
            S_z = pose[2] / 1000.0
            
            rad_Rx = (pose[3] / 360.0) * 2 * math.pi    # unit: radian
            rad_Ry = (pose[4] / 360.0) * 2 * math.pi
            rad_Rz = (pose[5] / 360.0) * 2 * math.pi
            point = [S_x, S_y, S_z, rad_Rx, rad_Ry, rad_Rz]

        if move_type == 'LINE_T':
            self.set_positions(SetPositionsRequest.LINE_T, point, speed, 0.5, blend_percent, True)
        elif move_type == 'PTP_T':
            self.set_positions(SetPositionsRequest.PTP_T, point, speed, 0.5, 100, True)
        elif move_type == 'PTP_J':
            self.set_positions(SetPositionsRequest.PTP_J, point, speed, 0.5, 100, True)
        else:
            raise ValueError("The move_type should be 'PTP_J', 'PTP_T', or 'LINE_T'.")
        
    def set_IO(self, module_name: str, pin: int, state: str = 'HIGH'):
        """
        Set default IO.

        :param module_name: 'endeffector' or 'controlbox'.
        :param pin: PIN value of module.
        :param state: the setup state. Default: 'HIGH'.
        """

        request = None
        if module_name == 'endeffector':
            request = SetIORequest.MODULE_ENDEFFECTOR
        elif module_name == 'controlbox':
            request = SetIORequest.MODULE_CONTROLBOX
        else:
            raise ValueError("The module_name should be 'endeffector' or 'controlbox'.")
        set_state = SetIORequest.STATE_OFF
        if state == 'HIGH':
            set_state = SetIORequest.STATE_ON
        self.set_io(request, SetIORequest.TYPE_DIGITAL_OUT, pin, set_state)
