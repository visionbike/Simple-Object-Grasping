from pathlib import Path
import numpy as np
import cv2


__all__ = ['CalibInfo', 'TransformInfo', 'Camera']


class CalibInfo:
    mat_cam = None
    mat_cam_new = None
    mat_cam_new_inv = None
    dist_coeff = None
    roi = None
    cx, cy = 0.0, 0.0
    fx, fy = 0.0, 0.0


class TransformInfo:
    mat_R = None
    mat_R_inv = None
    mat_Rt = None
    mat_P = None
    tvec1 = None
    rvec1 = None
    s = None


class Camera:
    def __init__(self, path_calib_info):
        self.path_calib_info = Path(path_calib_info)
        self.calib_info = self._load_calibration_info()
        self.transform_info = self._load_transformation_info()

    def _load_calibration_info(self):
        cf = CalibInfo()
        cf.mat_cam = np.load(str(self.path_calib_info / 'camera_matrix.npy'))
        cf.dist_coeff = np.load(str(self.path_calib_info / 'distortion_coeff.npy'))
        cf.mat_cam_new = np.load(str(self.path_calib_info / 'camera_matrix_new.npy'))
        cf.mat_cam_new_inv = np.load(str(self.path_calib_info / 'camera_matrix_new_inv.npy'))
        cf.roi = np.load(str(self.path_calib_info / 'roi.npy'))

        cf.cx = cf.mat_cam_new[0, 2]
        cf.cy = cf.mat_cam_new[1, 2]
        cf.fx = cf.mat_cam_new[0, 0]
        cf.fy = cf.mat_cam_new[1, 1]

        return cf

    def _load_transformation_info(self):
        tf = TransformInfo()
        tf.mat_R = np.load(str(self.path_calib_info / 'rotation_matrix.npy'))
        tf.mat_R_inv = np.linalg.inv(tf.mat_R)
        tf.mat_Rt = np.load(str(self.path_calib_info / 'extrinsic_matrix.npy'))
        tf.mat_P = np.load(str(self.path_calib_info / 'projection_matrix.npy'))
        tf.tvec1 = np.load(str(self.path_calib_info / 'tvec1.npy'))
        tf.rvec1 = np.load(str(self.path_calib_info / 'rvec1.npy'))
        tf.s = np.load(str(self.path_calib_info / 'scaling_factor.npy'))[0]

        return tf

    def calculate_XYZ(self, u, v):
        # Solve: From Image Pixels, find World Points
        uv1_ = np.array(
            [[u, v, 1]],
            dtype=np.float32
        ).T
        suv1_ = self.transform_info.s * uv1_
        xyz_c = self.calib_info.mat_cam_new_inv.dot(suv1_) - self.transform_info.tvec1
        XYZ_ = self.transform_info.mat_R_inv.dot(xyz_c)

        return XYZ_
