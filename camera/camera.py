from pathlib import Path
import numpy as np
import cv2

__all__ = ['Intrinsic', 'Extrinsic', 'Camera']


class Intrinsic:
    """
    Intrinsic Information
    """
    mat_cam = None
    mat_cam_new = None
    mat_cam_new_inv = None
    dist_coeff = None
    roi = None
    fx, fy = 0.0, 0.0
    cx, cy = 0.0, 0.0

    def __init__(self, path_calib_info: str, width: int, height: int):
        """

        :param path_calib_info: path that contain intrinsic information.
        :param width: the image width.
        :param height: the image height.
        """

        self.width = width
        self.height = height

        path_calib_info = Path(path_calib_info)
        assert path_calib_info.exists()

        self.mat_cam = np.load(str(path_calib_info / 'camera_matrix.npy'))
        self.dist_coeff = np.load(str(path_calib_info / 'distortion_coeff.npy'))

        # if using Alpha = 0, so we discard the black pixels from the distortion.
        # This helps make the entire region of interest is the full dimensions of the image (after `undistort`)
        # If using Alpha = 1, we retain the black pixels, and obtain the region of interest as the valid pixels for the matrix.
        # I will use Alpha = 1, so that I don't have to run undistort and can just calculate my real world x,y
        self.mat_cam_new, self.roi = cv2.getOptimalNewCameraMatrix(self.mat_cam, self.dist_coeff, (width, height), 1, (width, height))
        self.mat_cam_new_inv = np.linalg.inv(self.mat_cam_new)

        self.fx, self.fy = self.mat_cam_new[0, 0], self.mat_cam_new[1, 1]
        self.cx, self.cy = self.mat_cam_new[0, 2], self.mat_cam_new[1, 2]


class Extrinsic:
    tvec1 = None
    rvec1 = None
    scale = None
    mat_R = None
    mat_R_inv = None
    mat_Rt = None

    def __init__(self, path_calib_info: str):
        """

        :param path_calib_info: path that contain intrinsic information.
        """

        path_calib_info = Path(path_calib_info)
        assert path_calib_info.exists()

        self.rvec1 = np.load(str(path_calib_info / 'rvec1.npy'))
        self.tvec1 = np.load(str(path_calib_info / 'tvec1.npy'))
        self.scale = np.load(str(path_calib_info / 'scaling_factor.npy'))[0]

        self.mat_R, _ = cv2.Rodrigues(self.rvec1)
        self.mat_R_inv = np.linalg.inv(self.mat_R)
        self.mat_Rt = np.column_stack((self.mat_R, self.tvec1))


class Camera:
    def __init__(self, path_calib_info: str, width: int, height: int):
        """

        :param path_calib_info: path that contain intrinsic information.
        :param width: the image width.
        :param height: the image height.
        """
        self.intrinsic = Intrinsic(path_calib_info, width, height)
        self.extrinsic = Extrinsic(path_calib_info)

    def compute_XYZ(self, u: int, v: int):
        """

        :param u: the value along x-axis.
        :param v: the value along y-axis.
        :return: the real-world coordinate.
        """

        # solve: from image pixels,find world coordinate
        uv1_ = np.array([[u, v, 1]], dtype=np.float32).T
        suv1_ = self.extrinsic.scale * uv1_
        # get camera coordinates
        xyz_c = self.intrinsic.mat_cam_new_inv.dot(suv1_) - self.extrinsic.tvec1
        # get real-world coordinate
        XYZ_ = self.extrinsic.mat_R_inv.dot(xyz_c)
        return XYZ_

    def undistort_image(self, im: np.array):
        """
        Remove the distortion effect form image.

        :param im: the input image.
        :return: the undistorted image.
        """

        return cv2.undistort(im, self.intrinsic.mat_cam, self.intrinsic.dist_coeff, None, self.intrinsic.mat_cam_new)

    def preview_image(self, name: str, im: np.array, timeout: int = 0):
        """

        :param name: the window_name.
        :param im: the input image in OpenCV format.
        :param timeout: the timeout to close the window.
                    If timeout=0, the window will close when pressing ESC. Default: 0.
        """

        cv2.namedWindow(name, cv2.WINDOW_FULLSCREEN)
        cv2.imshow(name, im)
        cv2.waitKey(timeout)
        cv2.destroyAllWindows()
