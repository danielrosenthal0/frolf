import numpy as np 
import cv2
from imutils.video import FPS
import glob
num_corners = (6, 6)
square_size = 37.0
# Define the termination criteria for corner detection
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Create the object points for the calibration pattern
objp = np.zeros((num_corners[0]*num_corners[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:num_corners[0], 0:num_corners[1]].T.reshape(-1, 2)
objp *= square_size
# Create lists to store the object points and image points for each calibration image
objpoints = []
imgpoints = []
images = glob.glob("Calibration/*.jpg")
for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, num_corners, None)
    if ret:
        objpoints.append(objp)
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners)
        img = cv2.drawChessboardCorners(img, num_corners, corners, ret)
        cv2.imshow('img', img)
        cv2.waitKey(500)
# Calibrate the camera and save the calibration parameters
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
np.savetxt('camera_matrix.txt', mtx)
np.savetxt('distortion_coefficients.txt', dist)
# # Show an example of undistortion
# img = cv2.imread('calibration_images/example.jpg')
# h, w = img.shape[:2]
# newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
# dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
# cv2.imshow('img', img)
# cv2.imshow('dst', dst)
# cv2.waitKey(0)
# cv2.destroyAllWindows()