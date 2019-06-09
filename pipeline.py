import numpy as np
import cv2
from calibration import *
from helpers import *
from line import *

f = open('data.csv', 'w')

left_line_object = Line()
right_line_object = Line()

previous_out_image = None

def pipeline(image):
    # to track which approach was used to detect the lines
    found_by = ""

    # processed bird view
    out_image = None
    # previous processed bird view
    global previous_out_image

    # copy original image
    img = np.copy(image)

    # undistort image     
    undistorted_image = cv2.undistort(image, mtx, dist, None, mtx)
    # create "bird view"     
    transformed_image = transform_image(undistorted_image, M1)

    # use SobelX
    binary_output_sobelX = binary_output_sobel(transformed_image)

    # take a S channel
    binary_s = channel_threshold(hls_select(transformed_image, 2))

    # erode and dilate
    dilation = erode_and_dilate(binary_s | binary_output_sobelX)
    masked_transformed_image = cv2.bitwise_and(transformed_image, transformed_image, mask=dilation)

    # filter yellow and white colors     
    filtered_image = filter_line_colors(masked_transformed_image)
    # create gray image
    gray = cv2.cvtColor(filtered_image, cv2.COLOR_BGR2GRAY)

    # create binary image
    binary = np.zeros_like(gray)
    binary[gray > 0] = 1


    ploty = np.linspace(0, img.shape[0] - 1, img.shape[0])

    try:
        # try to find based lines based on the previous iteration
        if left_line_object.detected and right_line_object.detected:
            _lx, _rx, out_image, left_poly, right_poly = search_around_poly(binary, left_line_object.current_fit, right_line_object.current_fit)


            if not len(_lx) > 0 or not len(_rx) > 0:
                left_line_object.detected = False
                right_line_object.detected = False
            else:
                found_by = "search_around_poly"
                left_line_object.detected = True
                right_line_object.detected = True
    except:
        left_line_object.detected = False
        right_line_object.detected = False


    try:
        if left_line_object.detected == False or right_line_object.detected == False:
            _lx, _rx, out_image, left_poly, right_poly = fit_polynomial(binary)

            if not len(_lx) > 0 or not len(_rx) > 0:
                left_line_object.detected = False
                right_line_object.detected = False
            else:
                found_by = "fit_polynomial"
                left_line_object.detected = True
                right_line_object.detected = True
    except :
        left_line_object.detected = False
        right_line_object.detected = False

    if out_image is not None:
        previous_out_image = out_image
    else:
        out_image = previous_out_image

    # if nothing was found try to build it from the last matching
    if left_line_object.detected == False or right_line_object.detected == False:
        found_by = "not_found use previous data"
    else:

        # fit poly
        left_line_object.detected = True

        # remove suspicious poly, when detection works bad
        if abs(left_poly[0]) < 0.0007 and abs(left_poly[1]) < 0.8 and abs(left_poly[2]) < 1400:
            left_line_object.ally = ploty
            left_line_object.current_fit = left_poly
            left_line_object.allx = _lx
        else:
            text = "FAKE LEFT"
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(img, text, (10, 70), font, 0.7, (255, 0, 0), 2, cv2.LINE_AA)
            print(left_poly)

        right_line_object.detected = True

        if abs(right_poly[0]) < 0.0007 and abs(right_poly[1]) < 0.8 and abs(right_poly[2]) < 1400:
            right_line_object.ally = ploty
            right_line_object.current_fit = right_poly
            right_line_object.allx = _rx
        else:
            text = "FAKE RIGHT"
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(img, text, (100, 70), font, 0.7, (255, 0, 0), 2, cv2.LINE_AA)
            print(right_poly)

    try:
        f.write('{:10.5f}, {:10.5f}, {:10.5f},\n'.format(right_poly[0], right_poly[1], right_poly[2]))
    except:
        f.write('{:10.5f}, {:10.5f}, {:10.5f},\n'.format(0, 0, 0))

    # for debug purpose
    original_out = np.copy(out_image)

    result = draw_and_print_on_image(found_by, img, out_image)

    return result, transformed_image, original_out


def draw_and_print_on_image(found_by, img, out_image):
    # calcualte car position
    text = "CAR DISTANCE TO LEFT LINE : {:10.2f} CAR DISTANCE TO RIGHT LINE : {:10.2f}".format(
        left_line_object.measure_distance_real(out_image),
        right_line_object.measure_curvature_real(out_image)
    )
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, text, (10, 40), font, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    text = "Found by {0}".format(found_by)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, text, (10, 700), font, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    # left curv
    text = "Left curv: {:10.2f}".format(left_line_object.measure_curvature_real(out_image))
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, text, (10, 60), font, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    # right curv
    text = "Right curv: {:10.2f}".format(right_line_object.measure_curvature_real(out_image))
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, text, (660, 60), font, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    # poly params
    text = "Poly params: {:10.4f},{:10.4f},{:10.4f} |||| {:10.4f}, {:10.4f}, {:10.4f}".format(
        left_line_object.current_fit[0], left_line_object.current_fit[1], left_line_object.current_fit[2],
        right_line_object.current_fit[0], right_line_object.current_fit[1], right_line_object.current_fit[2])
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, text, (10, 630), font, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

    # collect points for lines and poly
    left_line_points = left_line_object.get_line_points()
    right_line_points = right_line_object.get_line_points()
    poly_points = np.concatenate((np.int32(left_line_points)[::-1], np.int32(right_line_points)))
    left_line_object.draw_line(out_image)
    right_line_object.draw_line(out_image)
    cv2.fillPoly(out_image, np.int32([poly_points]), (0, 255, 0))
    # convert "bird view" to normal
    normal_image = transform_image(out_image, M2)
    # Combine the result with the original image
    result = cv2.addWeighted(img, 1, normal_image, 0.6, 0)

    return result
