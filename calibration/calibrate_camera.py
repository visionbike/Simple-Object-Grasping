from pathlib import Path
import numpy as np
import cv2


CALIB_IMG_DIR = Path('calibration_images')
CALIB_INFO_DIR = Path('calibration_info')
CALIB_INFO_DIR.mkdir(parents=True, exist_ok=True)

# define the size in mm in chessboard
# MANUALLY EDIT
SQR_SIZE = 19

# define termination criteria
CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare grid of object points, like (0,0,0), (1,0,0), (2,0,0) ....,(7,7,0)
OBJ_IDX = np.zeros((7*7, 3), np.float32)
# add SQR_SIZE to account for SQR_SIZE cm per square in grid
OBJ_IDX[:, :2] = np.mgrid[0: 7, 0: 7].T.reshape(-1, 2) * SQR_SIZE

# arrays to store object points and image points from all images
obj_points = []     # 3d point in the real world space
img_points = []     # 2d points in the image plane

# define visualization window
window_name = 'Verify'
cv2.namedWindow(window_name, cv2.WINDOW_FULLSCREEN)

# define alpha argument for getting optimal new camera matrix after calibration
ALPHA = 1

if __name__ == '__main__':
    print('>>>> Reading calibration images...')
    found = 0
    for fname in CALIB_IMG_DIR.glob('*.jpg'):
        im = cv2.imread(str(fname))
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        # find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (7, 7), None)
        # if found, add object points, image points, (after refining them)
        if ret:
            found += 1
            obj_points.append(OBJ_IDX)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), CRITERIA)
            img_points.append(corners)
            # Draw and display the corners
            cv2.drawChessboardCorners(im, (7, 7), corners2, ret)
            cv2.imshow(window_name, im)
            cv2.waitKey(500)
    print(f'{found} images used for calibration')
    cv2.destroyAllWindows()

    print('>>>> Starting calibration...')
    ret, mat_cam, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, gray.shape[::-1], None, None)

    print('>>>> Verifying calibration...')

    # estimate re-projection error
    error_mean = 0.0
    for i in range(len(obj_points)):
        img_points_, mat_jacob = cv2.projectPoints(obj_points[i], rvecs[i], tvecs[i], mat_cam, dist)
        error_mean += cv2.norm(img_points[i], img_points_, cv2.NORM_L2) / len(img_points_)
    print(f'Re-projection error: {error_mean / len(obj_points)}')

    w, h = im.shape[:2]
    print(f'Image size (width, height): ({w}, {h})')
    # if using Alpha = 0, so we discard the black pixels from the distortion.
    # This helps make the entire region of interest is the full dimensions of the image (after `undistort`)
    # if using Alpha = 1, we retain the black pixels, and obtain the region of interest as the valid pixels for the matrix.
    # I will use Alpha = 1, so that I don't have to run undistort and can just calculate my real world x,y
    mat_newcam, roi = cv2.getOptimalNewCameraMatrix(mat_cam, dist, (w, h), ALPHA, (w, h))
    # get inverse new camera matrix
    mat_newcam_inverse = np.linalg.inv(mat_newcam)

    # undistort image
    im_undist = cv2.undistort(im, mat_newcam, dist, None, mat_newcam)

    cv2.imshow('Before', im)
    cv2.imshow('After', im_undist)
    cv2.waitKey(5000)
    cv2.destroyAllWindows()

    print('>>>> Saving calibration information...')
    print(f'Camera matrix:\n{mat_cam}')
    np.save(CALIB_INFO_DIR / 'camera_matrix.pny', mat_cam)
    print(f'Distortion Coefficients:\n{dist}')
    np.save(CALIB_INFO_DIR / 'distortion_coeff.pny', dist)
    print(f'Rotation vector:\n{rvecs}')
    np.save(CALIB_INFO_DIR / 'camera_rvecs.pny', rvecs)
    print(f'Translation vector:\n{dist}')
    np.save(CALIB_INFO_DIR / 'camera_tvecs.pny', tvecs)
    print(f'Region of Interest:\n{roi}')
    np.save(CALIB_INFO_DIR / 'roi.pny', roi)
    print(f'New camera matrix:\n{mat_newcam}')
    np.save(CALIB_INFO_DIR / 'camera_matrix_new.npy', mat_newcam)
    print(f'Inverse new camera matrix: {mat_newcam_inverse}')
    np.save(CALIB_INFO_DIR / 'camera_matrix_new_inv.npy', mat_newcam_inverse)
