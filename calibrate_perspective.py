from pathlib import Path
import numpy as np
import cv2
from camera import Intrinsic


# define calib info path
# MANUALLY EDIT
PATH_CALIB_INFO = './calibration_info'

# define the image points path
# MANUALLY EDIT
PATH_ENV_INFO = './environment_info'


# define the image center's real-world coordinates using the TM robot, value in mm
# MANUALLY EDIT
X_CENTER = 406.38
Y_CENTER = -283.65
Z_CENTER = 125.07


# define the pattern center's real-world coordinate using the TM robot, value in mm
# MANUALLY EDIT
WORLD_POINTS = [[X_CENTER, Y_CENTER, Z_CENTER],
                [463.92, -451.73, 123.57],	# point 0
                [378.83, -445.32, 122.63],	# point 1
                [273.73, -438.91, 121.68],	# point 2
                [468.61, -378.62, 121.32],	# point 3
                [373.59, -372.37, 122.35],	# point 4	
                [278.49, -365.91, 122.14],	# point 5
                [473.65, -305.31, 122.37],	# point 6
                [378.40, -299.21, 123.01],	# point 7	
                [283.04, -292.95, 122.68]]	# point 8

IMAGE_POINTS = None

if __name__ == '__main__':
    print('>>> Loading Calibration Info...')
    calib_info = Intrinsic(PATH_CALIB_INFO, 640, 480)

    print('>>>> Loading Image and Real-world Points...')
    WORLD_POINTS = np.array(WORLD_POINTS, dtype=np.float32)
    IMAGE_POINTS = np.load(str(Path(PATH_ENV_INFO) / 'image_coord.npy'))

    print('>>>> Solving PnP...')
    ret, rvec1, tvec1 = cv2.solvePnP(WORLD_POINTS, IMAGE_POINTS, calib_info.mat_cam_new, calib_info.dist_coeff)
    # compute rotation matrix
    mat_R, mat_jacob = cv2.Rodrigues(rvec1)
    # compute Rt matrix
    mat_Rt = np.column_stack((mat_R, tvec1))
    # compute projection matrix
    mat_P = calib_info.mat_cam_new.dot(mat_Rt)

    print('>>>> Estimate Scaling Factor...')
    s_arr = np.array([0], dtype=np.float32)
    s_describe = np.array([0] * len(WORLD_POINTS), dtype=np.float32)

    for i in range(0, len(WORLD_POINTS)):
        if i == 0:
            print(f'IMAGE CENTER')
        else:
            print(f'POINT #{i-1}')

        print("Forward: From World Points, Find Image Pixel...")
        XYZ1 = np.array(
            [[WORLD_POINTS[i, 0], WORLD_POINTS[i, 1], WORLD_POINTS[i, 2], 1]],
            dtype=np.float32
        ).T
        suv1 = mat_P.dot(XYZ1)
        s = suv1[2, 0]
        uv1 = suv1 / s
        print(f'Scaling Factor: {s}')
        s_arr = np.array([s / len(WORLD_POINTS) + s_arr[0]], dtype=np.float32)
        s_describe[i] = s

        print("Solve: From Image Pixels, find World Points...")
        uv1_ = np.array(
            [[IMAGE_POINTS[i, 0], IMAGE_POINTS[i, 1], 1]],
            dtype=np.float32
        ).T
        suv1_ = s * uv1_
        print("Get camera coordinates, multiply by inverse Camera Matrix, subtract tvec1")
        xyz_c = calib_info.mat_cam_new_inv.dot(suv1_) - tvec1
        mat_R_inv = np.linalg.inv(mat_R)
        XYZ_ = mat_R_inv.dot(xyz_c)
        print(f'XYZ: {XYZ_}')

    s_mean, s_std = np.mean(s_describe), np.std(s_describe)
    print('Result')
    print(f'Mean: {s_mean}')
    print(f'Std: {s_std}')
    print(">>>>>> S Error by Point")
    for i in range(len(WORLD_POINTS)):
        if i == 0:
            print(f'IMAGE CENTER')
        else:
            print(f'POINT #{i - 1}')
        print(f'S: {s_describe[i]} Mean: {s_mean} Error: {s_describe[i] - s_mean}')

    print('>>>> Saving to file....')

    np.save(str(Path(PATH_CALIB_INFO) / 'tvec1.npy'), tvec1)
    np.save(str(Path(PATH_CALIB_INFO) / 'rvec1.npy'), rvec1)
    np.save(str(Path(PATH_CALIB_INFO) / 'rotation_matrix.npy'), mat_R)
    np.save(str(Path(PATH_CALIB_INFO) / 'extrinsic_matrix.npy'), mat_Rt)
    np.save(str(Path(PATH_CALIB_INFO) / 'projection_matrix.npy'), mat_P)
    np.save(str(Path(PATH_CALIB_INFO) / 'scaling_factor.npy'), s_arr)
