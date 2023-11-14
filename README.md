# Simple Object Grasping with TM Robot Arm

## Preparation

You should set up the working environment following the assumption that the working space's origin is robot's working and the robot's end-effector and camera should be perpendicular to the
working space surface. 

Install ROS and TM Driver.

Create ROS working space.

Install Python3 and OpenCV4.

Download the source code from [https://github.com/visionbike/Simple-Object-Grasping.git](https://github.com/visionbike/Simple-Object-Grasping.git)

```shell
git clone https://github.com/visionbike/Simple-Object-Grasping.git
```

## Camera Calibration

First, you need to prepare the images for camera calibration. The images will be store in the `calibration_images` folder.

```shell
python3 capture_calib_images.py
```

Run `calibrate_camera.py` to obtain the camera's intrinsic matrix and distortion coefficients.

```shell
python3 calibrate_camera.py
```

There information will be stored as '*.npy' files in `calibration_info` folder. The re-projection error should be **<0.25**
for better results.

## Perspective Calibration (3D Construction)

The extrinsic matrix **R|t** can be obtained by solving Perspective-n-Point problem for estimate the posed of the calibrated camera
given a set of 3D points in the world and their corresponding 2D projections in the image.

To get 2D points im the image, run `detect_points.py`. We apply the image difference between the background image working space 
and the image that identify 9 circle patterns. The method simply detect the external contours of the 9 circle patterns then extract 
the corresponding centroids. You can **adjust** `MIN_AREA`, `MAX_AREA`, `MIN_ASPECT_RATIO`, `MAX_ASPECT_RATIO` values to eliminate 
the detected corner points and non-defined circle patterns.

```shell
python3 detect_points.py
```

The detected centroids and the image center are saved in `envinronment_info` folder. Open the `image_coord.png` to see the order the centroids. 

You use the TM Robot Arm to detect the corresponding world coordinate, 
assuming that the working space's origin is robot's working and the robot's end-effector and camera should be perpendicular to the
working space's surface. You need to collect the 3D coordinate following the order of image center, centroid 0,centroid 1, ..., centroid 8. 
The 3D coordinate of detected points need to be added to `calibrate_perspective.py`.

Run `calibrate_perspective.py` to obtain the extrinsic matrix `R|t` and scaling factor `s`. The standard deviation of `s` should be **<5mm**.

```shell
python3 calibrate_perspective.py
```

For verification, run `verify_xyz_estimation.py`. The code support to click a pixel on monitor to get the image point, and command the TM robot 
arm to move to the 3D point estimated from the clicked point. 

Note that you need to detect the Z value of the working space's surface by TM robot 
arm and modify `Z_WORKING_SPACE` in the `verify_xyz_estimation.py`. You also need to define the initial 3D point for the robot to start do the 
action by modify `WORLD_POINT_INIT` (just change the first three values which stand for X, Y, Z).

```shell
python3 verify_xyz_estimation.py
```

For more precise 3D coordinate estimation, retry the calibration process many times to save the best result.

## Object Grasping Via Object Shape

The method using the image difference method between the background Region of Interest (ROI) - **bg** image and the ROI that contains objects - 
**fg** image. The **bg** and **fg** images are converted into gray-scale images, then the different image is computed by subtracting these images.

```python
def compute_image_difference(bg, fg, min_thresh, max_thresh, sensitivity):
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
```

The objects will be highlighted in the resulted image. The Gaussian blur is applied to removed possible noise (small regions), which
would affect the detection result.

The object's is detected by the finding external contour. The contour candidates will be filtered by the contour's area. This helps to 
remove unexpected contour or possible corners. Please adjust the `MIN_AREA`. `MAX_AREA`, `OTSU_SENSITIVITY`, `OTSU_LOW_THRESH` and `OTSU_HIGH_THRESH`
based on your objects.

```python
# define valid contour parameter limits in pixels
# MANUALLY EDIT
MIN_AREA = 200  # 10 x 10
MAX_AREA = 10000  # 100 x 100

# define thresholds for Otsu method
# MANUALLY EDIT
OTSU_SENSITIVITY = 22
OTSU_LOW_THRESH = 45
OTSU_HIGH_THRESH = 255
```

The centroid and the angle of the detected contour is computed based on `cv2.minAreaRect`.

```python
def get_valid_contours(contours, min_area, max_area):
    valid_ids = []
    for i, contour in enumerate(contours):
        contour_area = cv2.contourArea(contour)

        # discard contours that out of range in terms of contour area
        if min_area < contour_area < max_area:
            valid_ids.append(i)
    return valid_ids


def detect_contours(bg, fg, min_thresh, max_thresh, sensitivity, min_area, max_area):
    # compute image difference
    diff = compute_image_difference(bg, fg, min_thresh, max_thresh, sensitivity)

    # find the contours
    # use RETR_EXTERNAL for only outer contours
    # use RETR_TREE for all the hierarchy
    contours, hierarchy = cv2.findContours(diff, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # identify the valid contours
    contour_ids = get_valid_contours(contours, min_area, max_area)

    # detect centroids, including
    # (cx, cy): centroid point
    # (w, h): width, height of min area rectangle bounding box
    # angle: the angle to detect the direction
    centroids = []
    for i, idx in enumerate(contour_ids):
        contour = contours[idx]

        # get centroid and angle
        centroid, size, angle = cv2.minAreaRect(contour)

        # estimate centroid of contours
        M = cv2.moments(contour)
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])

        # [(cx, cy), (w, h), angle]
        centroids.append([[cx, cy], list(size), angle])
    return contours, centroids
```

The centroids and corresponding angle will then are used to point out the 3D position and the orientation for TM robot arm to grasp objects.
The `TMRobot` class, which is defined in `./robot_control/tm_robot.py`, uses the API provide by TM Driver to control robot.
You need to define `WORLD_POINT_INIT` value to specify the starting position that the robot is ready to do the grasping task.

```python
# define initial world point
# MANUALLY EDIT
# [X, Y, Z, RX, RY, RZ]
WORLD_POINT_INIT = [-35.6, -421.65, 244.2, 180, 0, 90]
```

You also need to define the Z value the working space and the offset to make sure the end-effector not to touch the working
space surface's, leading the robot's termination. The offset can be adjusted based on the height of the object.

```python
# define working space world Z in mm
# MANUALLY EDIT
Z_WORKING_SPACE = 128

# define the offset world Z in mm
# the length of the tcp point
# MANUALLY EDIT
Z_OFFSET = -23
```

The function to define the TM robot arm's movement will be defined in `control_TM_arm()` function.

```python
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
        rospy.sleep(3)  # unit: s
        # open the gripper
        tm_robot.set_IO('endeffector', 0, state='LOW')
        rospy.sleep(3)  # unit: s
        # move to the intermediate position
        tm_robot.move(world_point_inter, move_type='PTP_T', speed=2.5, blend_mode=False)
        rospy.sleep(3)  # unit: s
        # move to the end position
        tm_robot.move(world_point_end, move_type='PTP_T', speed=2.5, blend_mode=False)
        rospy.sleep(3)  # unit: s
        # close the gripper
        tm_robot.set_IO('endeffector', 0, state='HIGH')
        rospy.sleep(1)  # unit: s
        # move back to the initial position
        tm_robot.move(world_point_init, move_type='PTP_T', speed=2.5, blend_mode=False)
        rospy.sleep(3)  # unit: s
        # close the gripper
        tm_robot.set_IO('endeffector', 0, state='LOW')
```
