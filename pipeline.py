import numpy as np
import cv2
from calibration import *
from helpers import *


def pipeline(image):
    # copy original image     
    img = np.copy(image)

    # undistort image     
    undistorted_image = cv2.undistort(image, mtx, dist, None, mtx)

    # region
    # 7. define a region of interest
    ysize = 1200
    xsize = 720
    left_bottom = [0, ysize]
    right_bottom = [xsize, ysize]
    apex_d = 30
    apex_1 = [72 / 2 + apex_d, ysize * 0.59]
    apex_2 = [xsize / 2 - apex_d, ysize * 0.59]

    # 8. select region - trapezoid
    region = region_of_interest(undistorted_image, np.array([[[left_bottom, right_bottom, apex_1, apex_2]]], 'int32'))


    # create "bird view"     
    transformed_image = transform_image(undistorted_image, M1)
    # filter yellow and white colors     
    filtered_image = filter_line_colors(transformed_image)
    # create gray image
    gray = cv2.cvtColor(filtered_image, cv2.COLOR_BGR2GRAY)

    # create binary image
    binary = np.zeros_like(gray)
    binary[gray > 0] = 1

    try:
        # fit poly     
        left_line, right_line, out_image, left_poly, right_poly = fit_polynomial(binary)
    except TypeError as e:
        return image, image, image

    # for debug purpose     
    original_out = np.copy(out_image)

    # calcualte car position
    f_l = np.poly1d(left_poly)
    f_r = np.poly1d(right_poly)
    middle = img.shape[1] / 2
    text = "CAR DISTANCE TO LEFT LINE : {0} CAR DISTANCE TO RIGHT LINE : {1}".format(middle - f_l(img.shape[1]),
                                                                                     f_r(img.shape[1]) - middle)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, text, (10, 30), font, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

    # draw lines
    ploty = np.linspace(0, img.shape[0] - 1, img.shape[0])

    # collect points for lines and poly  
    left_line_points = np.dstack((left_line, ploty))[0]
    right_line_points = np.dstack((right_line, ploty))[0]

    poly_points = np.concatenate((np.int32(left_line_points)[::-1], np.int32(right_line_points)))

    cv2.polylines(out_image, np.int32([left_line_points]), False, (255, 150, 0), 140)
    cv2.polylines(out_image, np.int32([right_line_points]), False, (0, 150, 255), 140)
    cv2.fillPoly(out_image, np.int32([poly_points]), (0, 255, 0))

    # convert "bird view" to normal
    normal_image = transform_image(out_image, M2)

    # Combine the result with the original image
    result = cv2.addWeighted(img, 1, normal_image, 0.6, 0)

    return result, transformed_image, original_out
