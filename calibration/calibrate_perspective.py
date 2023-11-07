from pathlib import Path
import numpy as np
import cv2


class CalibInfo:
    mat_cam = None
    mat_cam_new = None
    mat_cam_new_inv = None
    dist_coeff = None
    roi = None
    cx, cy = 0.0, 0.0
    fx, fy = 0.0, 0.0


def load_calibration_info(path_calib_info):
    cf = CalibInfo()
    cf.mat_cam = np.load(str(path_calib_info / 'camera_matrix.npy'))
    cf.dist_coeff = np.load(str(path_calib_info / 'distortion_coeff.npy'))
    cf.mat_cam_new = np.load(str(path_calib_info / 'camera_matrix_new.npy'))
    cf.mat_cam_new_inv = np.load(str(path_calib_info / 'camera_matrix_new_inv.npy'))
    cf.roi = np.load(str(path_calib_info / 'roi.npy'))

    cf.cx = CalibInfo.mat_cam_new[0, 2]
    cf.cy = CalibInfo.mat_cam_new[1, 2]
    cf.fx = CalibInfo.mat_cam_new[0, 0]
    cf.fy = CalibInfo.mat_cam_new[1, 1]

    return cf


# define calib info path
# MANUALLY EDIT
PATH_CALIB_INFO = Path('calibration_info')

# define the image points path
# MANUALLY EDIT
PATH_ENV_INFO = Path('environment_info')


# define the image center's real-world coordinates using the TM robot, value in mm
# MANUALLY EDIT
X_CENTER = 109
Y_CENTER = 107
Z_CENTER = 234


# define the pattern center's real-world coordinate using the TM robot, value in mm
# MANUALLY EDIT
WORLD_POINTS = [[X_CENTER, Y_CENTER, Z_CENTER],
                [55, 39, 268],
                [142, 39, 270],
                [228, 39, 274],
                [55, 106, 242],
                [142, 106, 238],
                [228, 106, 248],
                [55, 173, 230],
                [142, 173, 225],
                [228, 173, 244]]

IMAGE_POINTS = None

if __name__ == '__main__':
    print('>>> Loading Calibration Info...')
    calib_info = load_calibration_info(PATH_CALIB_INFO)

    print('>>>> Loading Image and Real-world Points...')
    WORLD_POINTS = np.array(WORLD_POINTS, dtype=np.float32)
    IMAGE_POINTS = np.load(str(PATH_ENV_INFO / 'image_coord.npy'))

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
        print(f'POINT #{i}')

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
        print(f'Point {i}')
        print(f'S: {s_describe[i]} Mean: {s_mean} Error: {s_describe[i] - s_mean}')

    print('>>>> Saving to file....')
    np.save(str(PATH_CALIB_INFO / 'extrinsic_matrix.npy'), mat_Rt)
    np.save(str(PATH_CALIB_INFO / 'projection_matrix.npy'), mat_P)
    np.save(str(PATH_CALIB_INFO / 'scaling_factor.npy'), s_arr)
