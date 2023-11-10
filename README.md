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

There information will be stored as '*.npy' files in `calibration_info` folder. The re-projection error should be **<0.25**
for better results.

## Perspective Calibration (3D Construction)

The extrinsic matrix **R|t** can be obtained by solving Perspective-n-Point problem for estimate the posed of the calibrated camera
given a set of 3D points in the world and their corresponding 2D projections in the image.

To get 2D points im the image, run `detect_points.py`. I apply the image difference between the background image working space 
and the image that contain 9 circle patterns. The method simply detect the external contours of the 9 circle patterns then extract 
the corresponding centroids. You can **adjust** `MIN_AREA`, `MAX_AREA`, `MIN_RATIO`, `MAX_RATION` values to eliminate the detected 
corner points and non-defined circle patterns.

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
