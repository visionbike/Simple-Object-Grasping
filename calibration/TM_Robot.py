import rospy
from tm_msgs.msg import *
from tm_msgs.srv import *
import math


class TM_Robot():
    rospy.wait_for_service('tm_driver/set_positions')
    rospy.wait_for_service('tm_driver/set_io')
    set_positions = rospy.ServiceProxy('tm_driver/set_positions', SetPositions)
    set_io = rospy.ServiceProxy('tm_driver/set_io', SetIO)

    def __init__(self, pose):
        self.Init_Pose = pose

    def move_TM(self, pose, Move_type='PTP_J', Speed=1.0, blend_Mode=True):
        blend_percent = 0
        print("Moving to next point...")

        if blend_Mode:
            blend_percent = 100
        if Move_type == 'PTP_J':
            J1 = (pose[0] / 180.0) * math.pi            # unit: radian
            J2 = (pose[1] / 180.0) * math.pi
            J3 = (pose[2] / 180.0) * math.pi
            J4 = (pose[3] / 180.0) * math.pi
            J5 = (pose[4] / 180.0) * math.pi
            J6 = (pose[5] / 180.0) * math.pi
            point_J = [J1, J2, J3, J4, J5, J6]
        else:
            S_x = pose[0] / 1000.0                      # unit: mm
            S_y = pose[1] / 1000.0
            S_z = pose[2] / 1000.0
            
            rad_Rx = (pose[3] / 360.0) * 2 * math.pi    # unit: radian
            rad_Ry = (pose[4] / 360.0) * 2 * math.pi
            rad_Rz = (pose[5] / 360.0) * 2 * math.pi
            point_C = [S_x, S_y, S_z, rad_Rx, rad_Ry, rad_Rz]

        print("Type: " + Move_type)
            
        if Move_type == 'LINE_T':
            self.set_positions(SetPositionsRequest.LINE_T, point_C, Speed, 0.5, blend_percent, True)
        elif Move_type == 'PTP_T':
            self.set_positions(SetPositionsRequest.PTP_T, point_C, Speed, 0.5, 100, True)
        elif Move_type == 'PTP_J':
            self.set_positions(SetPositionsRequest.PTP_J, point_J, Speed, 0.5, 100, True)
        else:
            print("The Move_type should be 'PTP_J', 'PTP_T', or 'LINE_T'")
        
    def set_IO(self, module_name, pin, state='HIGH'):
        if module_name == 'endeffector':
            request = SetIORequest.MODULE_ENDEFFECTOR
        elif module_name == 'controlbox':
            request = SetIORequest.MODULE_CONTROLBOX
        set_state = SetIORequest.STATE_OFF
        if state == 'HIGH':
            set_state = SetIORequest.STATE_ON
        self.set_io(request, SetIORequest.TYPE_DIGITAL_OUT, pin, set_state)
