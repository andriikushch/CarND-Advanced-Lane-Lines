import glob
import numpy as np
import cv2
import matplotlib.image as mpimg


# read images for calibration
calibration_image_list = glob.glob('camera_cal/calibration*.jpg')

# images with corners
images_with_corners = {}

# points collections
objpoints = []
imgpoints = []

nx_9_ny_6 = [
    'camera_cal/calibration2.jpg',
    'camera_cal/calibration3.jpg',
    'camera_cal/calibration6.jpg',
    'camera_cal/calibration7.jpg',
    'camera_cal/calibration8.jpg',
    'camera_cal/calibration9.jpg',
    'camera_cal/calibration10.jpg',
    'camera_cal/calibration11.jpg',
    'camera_cal/calibration12.jpg',
    'camera_cal/calibration13.jpg',
    'camera_cal/calibration14.jpg',
    'camera_cal/calibration15.jpg',
    'camera_cal/calibration16.jpg',
    'camera_cal/calibration17.jpg',
    'camera_cal/calibration18.jpg',
    'camera_cal/calibration19.jpg',
    'camera_cal/calibration20.jpg',
]

nx_9_ny_5 = [
    'camera_cal/calibration1.jpg'
]

nx_7_ny_6 = [
    'camera_cal/calibration5.jpg'
]

nx_5_ny_7 = [
    'camera_cal/calibration4.jpg'
]

for file_name in calibration_image_list:
    # nx corners in x dimension
    # ny corners in y dimension
    if file_name in nx_9_ny_6:
        nx = 9
        ny = 6
    elif file_name in nx_9_ny_5:
        nx = 9
        ny = 5
    elif file_name in nx_7_ny_6:
        nx = 7
        ny = 6
    elif file_name in nx_5_ny_7:
        nx = 5
        ny = 7
    else:
        raise Exception("unknown file name ", file_name)

    # prepare object points
    objp = np.zeros((nx * ny, 3), np.float32)
    objp[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)

    # read image
    image = mpimg.imread(file_name)

    # conver to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # try to find a corners
    ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

    # if found, draw corners
    if ret == True:
        imgpoints.append(corners)
        objpoints.append(objp)

        # draw and display the corners
        cv2.drawChessboardCorners(image, (nx, ny), corners, ret)
        images_with_corners[file_name] = image
    else:
        print(file_name, nx, ny)

# calibrate the camera
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
