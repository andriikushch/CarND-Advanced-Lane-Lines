import numpy as np
import cv2
import matplotlib.pyplot as plt


# display image map
def plot_images_map(images, img_size=(20, 15), columns=5):
    plt.figure(figsize=img_size)

    i = 0
    for file_name in images:
        plt.subplot(len(images) / columns + 1, columns, i + 1).set_title(file_name)
        plt.imshow(images[file_name])
        i += 1


# return one of HLS chanels
def hls_select(img, channel=0):
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    return hls[:, :, channel]


# filter white and yellow lines
def filter_line_colors(img):
    # conver image to HSV, for easier color selection
    # it is necessary step in this pipeline for an optional chalange
    # to get rid of errors that are introduced by the mirrored hood
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # set white color boundaries, values are estimated experimentaly
    lower_white = np.array([100, 80, 180])
    upper_white = np.array([255, 255, 255])

    # threshold the HSV image to get only white colors
    white_mask = cv2.inRange(img, lower_white, upper_white)

    # set yellow color boundaries, values are estimated experimentaly
    lower_yellow = np.array([50, 50, 50])
    upper_yellow = np.array([110, 255, 255])

    # threshold the HSV image to get only yellow colors
    yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

    # use mask on image and return
    return cv2.bitwise_and(img, img, mask=white_mask | yellow_mask)


# def region_of_interest(img, vertices):
#     # defining a blank mask to start with
#     mask = np.zeros_like(img)
#
#     # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
#     if len(img.shape) > 2:
#         channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
#         ignore_mask_color = (255,) * channel_count
#     else:
#         ignore_mask_color = 255
#
#     # filling pixels inside the polygon defined by "vertices" with the fill color
#     cv2.fillPoly(mask, vertices, ignore_mask_color)
#
#     # returning the image only where mask pixels are nonzero
#     masked_image = cv2.bitwise_and(img, mask)
#     return masked_image


# function for calculating transform matrix for a bird view
def calculate_transform_matrix(forward=True):
    # manually defined points
    # src = np.float32([
    #     [596, 450],
    #     [685, 450],
    #     [1020, 662],
    #     [295, 662]
    # ])
    #
    # dst = np.float32([
    #     [200, 50],
    #     [1024, 50],
    #     [1024, 720],
    #     [200, 720]
    # ])

    # src = np.float32([
    #     [200, 720],
    #     [608, 440],
    #     [675, 440],
    #     [1100, 720]
    # ])
    #
    # dst = np.float32([
    #     [200, 720],
    #     [200, 0],
    #     [1100, 0],
    #     [1100, 720]
    # ])

    # src = np.float32([
    #     [581, 477],
    #     [699, 477],
    #     [896, 675],
    #     [384, 675]
    # ])
    #
    # dst = np.float32([
    #     [384, 0],
    #     [896, 0],
    #     [896, 720],
    #     [384, 720]
    # ])

    src = np.float32([[580, 460], [205, 720], [1110, 720], [703, 460]])
    dst = np.float32([[320, 0], [320, 720], [960, 720], [960, 0]])

    # xxx = 588
    #
    # # yyy = 430
    #
    # yyy = 445
    # src = np.float32([
    #     [xxx, yyy],  #
    #     [1280 - xxx, yyy],  #
    #     [1280, 720],
    #     [0, 720]
    # ])
    #
    # dst = np.float32([
    #     [0, 0],
    #     [1280, 0],
    #     [1280, 720],
    #     [0, 720]
    # ])
    if forward:
        return cv2.getPerspectiveTransform(src, dst)

    return cv2.getPerspectiveTransform(dst, src)


# forward matrix transformation
M1 = calculate_transform_matrix()

# backward matrix transformation
M2 = calculate_transform_matrix(False)


# warpPerspective function wrapper
def transform_image(img, M):
    warped = cv2.warpPerspective(
        img,
        M, (img.shape[1], img.shape[0]),
        flags=cv2.WARP_FILL_OUTLIERS + cv2.INTER_CUBIC)

    return warped


# direct search using iteration with window
def find_lane_pixels(binary_warped):
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0] // 2:, :], axis=0)
    # Create an output image to draw on and visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0] // 2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # HYPERPARAMETERS
    # Choose the number of sliding windows
    nwindows = 20
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50

    # Set height of windows - based on nwindows above and image shape
    window_height = np.int(binary_warped.shape[0] // nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated later for each window in nwindows
    leftx_current = leftx_base
    rightx_current = rightx_base

    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window + 1) * window_height
        win_y_high = binary_warped.shape[0] - window * window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin

        # Draw the windows on the visualization image
        cv2.rectangle(out_img, (win_xleft_low, win_y_low),
                      (win_xleft_high, win_y_high), (0, 255, 0), 6)

        cv2.rectangle(out_img, (win_xright_low, win_y_low),
                      (win_xright_high, win_y_high), (0, 255, 0), 6)

        # Identify the nonzero pixels in x and y within the window #
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                          (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                           (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices (previously was a list of lists of pixels)
    try:
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
    except ValueError:
        # Avoids an error if the above is not implemented fully
        pass

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    return leftx, lefty, rightx, righty, out_img


def fit_polynomial(binary_warped):
    # Find our lane pixels first
    leftx, lefty, rightx, righty, out_img = find_lane_pixels(binary_warped)

    # Fit a second order polynomial to each using `np.polyfit`
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
    try:
        left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
        right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]
    except TypeError:
        # Avoids an error if `left` and `right_fit` are still none or incorrect
        print('The function failed to fit a line!')
        left_fitx = 1 * ploty ** 2 + 1 * ploty
        right_fitx = 1 * ploty ** 2 + 1 * ploty

    ## Visualization ##
    # Colors in the left and right lane regions
    out_img[lefty, leftx] = [255, 0, 0]
    out_img[righty, rightx] = [0, 0, 255]

    return left_fitx, right_fitx, out_img, left_fit, right_fit


def fit_poly(img_shape, leftx, lefty, rightx, righty):
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    # Generate x and y values for plotting
    ploty = np.linspace(0, img_shape[0] - 1, img_shape[0])

    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

    return left_fitx, right_fitx, ploty, left_fit, right_fit


# try to reuse data from previous step
def search_around_poly(binary_warped, left_fit, right_fit):
    # margin around the previous polynomial to search
    margin = 100

    # Grab activated pixels
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    left_lane_inds = ((nonzerox > (left_fit[0] * (nonzeroy ** 2) + left_fit[1] * nonzeroy +
                                   left_fit[2] - margin)) & (nonzerox < (left_fit[0] * (nonzeroy ** 2) +
                                                                         left_fit[1] * nonzeroy + left_fit[
                                                                             2] + margin)))
    right_lane_inds = ((nonzerox > (right_fit[0] * (nonzeroy ** 2) + right_fit[1] * nonzeroy +
                                    right_fit[2] - margin)) & (nonzerox < (right_fit[0] * (nonzeroy ** 2) +
                                                                           right_fit[1] * nonzeroy + right_fit[
                                                                               2] + margin)))

    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit new polynomials
    left_fitx, right_fitx, ploty, left_fit, right_fit = fit_poly(binary_warped.shape, leftx, lefty, rightx, righty)

    ## Visualization ##
    # Create an image to draw on and an image to show the selection window
    out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255

    # Color in left and right line pixels
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

    return left_fitx, right_fitx, out_img, left_fit, right_fit


def channel_threshold(channel, min=120, max=255):
    binary = np.zeros_like(channel)
    binary[(channel > min) & (channel <= max)] = 1

    return binary


def erode_and_dilate(binary):
    kernel = np.ones((3, 3), np.uint8)
    erosion = cv2.erode(binary, kernel, iterations=1)

    kernel = np.ones((12, 12), np.uint8)
    dilation = cv2.dilate(erosion, kernel, iterations=2)
    return dilation


def binary_output_sobel(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    kernel = np.ones((3, 3), np.float32) / 25
    dst = cv2.filter2D(gray, -1, kernel)
    sobelx = cv2.Sobel(dst, cv2.CV_64F, 1, 0, ksize=5)
    abs_sobel_x = np.absolute(sobelx)

    scaled_sobel = np.uint8(255 * abs_sobel_x / np.max(abs_sobel_x))
    binary_output_sobel = np.zeros_like(scaled_sobel)
    binary_output_sobel[(scaled_sobel >= 45) & (scaled_sobel <= 120)] = 1

    return  binary_output_sobel
