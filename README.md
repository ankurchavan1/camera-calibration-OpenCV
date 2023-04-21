# Camera Projection and Calibration

This project tackles two fundamental problems in computer vision - camera projection and calibration, using Python and two powerful libraries - NumPy and OpenCV.

## Dependencies

- Python 
- NumPy 
- OpenCV 

## Problem 1 - Camera Projection

Given a camera matrix, the goal is to find the intrinsic and extrinsic parameters of the camera. This is accomplished by decomposing the projection matrix into its constituent parts using NumPy's `linalg` module.

## Problem 2 - Camera Calibration

Camera calibration is the process of estimating the intrinsic and extrinsic parameters of a camera given a set of images. In this problem, 13 images of a chessboard from different angles, distances, and viewpoints are used. Corresponding 2D image points and 3D world points on the chessboard are found and used to calibrate the camera. The reprojection error is calculated to evaluate the accuracy of the calibration matrix.

OpenCV's `calibrateCamera` function is used to perform the calibration. Various ways to increase the accuracy of the calibration matrix are discussed, such as using more images, choosing an appropriate calibration pattern, and improving the accuracy of the detected corner points.

## Conclusion

By using NumPy and OpenCV, these two fundamental problems in computer vision - camera projection and calibration - have been successfully tackled. This project provides a good starting point for those interested in computer vision and image processing.
