import numpy as np
import cv2
from calibration import *
from line import *
import matplotlib.pyplot as plt

class LineDetector:
    def __init__(self):
        self.left_line_object = Line()
        self.right_line_object = Line()
        # var to track which approach was used to detect the lines
        self.found_by = ""

        # forward matrix transformation
        self.M1 = self.calculate_transform_matrix()
        # backward matrix transformation
        self.M2 = self.calculate_transform_matrix(False)

    def pipeline(self, _img):
        # 1. Copy the original image
        img = np.copy(_img)

        # 2. Undistort image
        undistorted_image = cv2.undistort(_img, mtx, dist, None, mtx)

        # 3. Create "bird view" from undistorted_image
        transformed_image = self.transform_image(undistorted_image, self.M1)

        # 4. Apply SobelX operator to "transformed_image"
        binary_output_sobel_x = self.binary_output_sobel(transformed_image)

        # 5. Take a S channel of transformed_image
        binary_s = self.channel_threshold(self.hls_select(transformed_image, 2))

        # 6. "Erode" and "Dilate" the logical or between s-channel and sobelX images
        dilation = self.erode_and_dilate(binary_s | binary_output_sobel_x)

        # 7. Mask transformed image
        masked_transformed_image = cv2.bitwise_and(transformed_image, transformed_image, mask=dilation)

        # 8. Filter yellow and white colors on masked transformed image
        filtered_image = self.filter_line_colors(masked_transformed_image)

        # 9. Create gray image to create binary image
        gray = cv2.cvtColor(filtered_image, cv2.COLOR_BGR2GRAY)

        # 10. Create binary image out of gray
        binary = np.zeros_like(gray)
        binary[gray > 0] = 1

        # try to find the lines, first with quick search if lines were already detected
        _lx, _rx, left_poly, out_image, right_poly = None, None, None, None, None,
        try:
            _lx, _rx, left_poly, out_image, right_poly = self.try_to_find_points(binary, "search_around_poly",
                                                                                 self.search_around_poly)
        except:
            self.left_line_object.detected = False
            self.right_line_object.detected = False

        # long search if "search_around_poly" search fail
        try:
            _lx, _rx, left_poly, out_image, right_poly = self.try_to_find_points(binary, "fit_polynomial",
                                                                                 self.fit_polynomial)
        except:
            self.left_line_object.detected = False
            self.right_line_object.detected = False

        ploty = np.linspace(0, img.shape[0] - 1, img.shape[0])

        if self.left_line_object.detected and self.right_line_object.detected:
            self.left_line_object.detected = True
            self.left_line_object.ally = ploty
            self.left_line_object.current_fit = left_poly
            self.left_line_object.allx = _lx

            self.right_line_object.detected = True
            self.right_line_object.ally = ploty
            self.right_line_object.current_fit = right_poly
            self.right_line_object.allx = _rx

            result = self.format_result(img, out_image)

            return result, undistorted_image, transformed_image, binary_output_sobel_x, binary_s, dilation, masked_transformed_image, filtered_image, gray, binary, out_image
        else:
            return img, img, img, img, img, img, img, img, img, img, img

    @staticmethod
    def calculate_transform_matrix(forward=True):
        '''
        Calculates transform matrix for a bird view
        '''
        # manually defined points
        src = np.float32([[580, 460], [204, 720], [1110, 720], [702, 460]])
        dst = np.float32([[320, 0], [320, 720], [960, 720], [960, 0]])

        if forward:
            return cv2.getPerspectiveTransform(src, dst)

        return cv2.getPerspectiveTransform(dst, src)

    def try_to_find_points(self, binary, tag, method):
        '''
        Tries to find a line parameters based on provided method
        '''
        if self.left_line_object.detected == False or self.right_line_object.detected == False:
            _lx, _rx, out_image, left_poly, right_poly = method(binary)

            if not len(_lx) > 0 or not len(_rx) > 0:
                self.left_line_object.detected = False
                self.right_line_object.detected = False
            else:
                self.found_by = tag
                self.left_line_object.detected = True
                self.right_line_object.detected = True
        return _lx, _rx, left_poly, out_image, right_poly

    def format_result(self, img, out_image):
        '''
        Transform image back and print info
        '''

        # calculate car position

        # sum of distance to left and right lines
        lane_width = self.left_line_object.measure_distance_real(out_image) + self.right_line_object.measure_distance_real(out_image)
        bigger_distance = max([self.left_line_object.measure_distance_real(out_image), self.right_line_object.measure_distance_real(out_image)])

        text = "CAR DISTANCE TO THE CENTER : {:10.2f} m".format(abs(bigger_distance-lane_width/2))
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, text, (10, 40), font, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        text = "Found by {0}".format(self.found_by)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, text, (10, 700), font, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

        # left curv
        text = "Left curv: {:10.2f}".format(self.left_line_object.measure_curvature_real(out_image))
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, text, (10, 60), font, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

        # right curv
        text = "Right curv: {:10.2f}".format(self.right_line_object.measure_curvature_real(out_image))
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, text, (660, 60), font, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

        # poly params
        text = "Poly params: {:10.4f},{:10.4f},{:10.4f} |||| {:10.4f}, {:10.4f}, {:10.4f}".format(
            self.left_line_object.current_fit[0], self.left_line_object.current_fit[1],
            self.left_line_object.current_fit[2],
            self.right_line_object.current_fit[0], self.right_line_object.current_fit[1],
            self.right_line_object.current_fit[2])
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, text, (10, 630), font, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

        # collect points for lines and poly
        left_line_points = self.left_line_object.get_line_points()
        right_line_points = self.right_line_object.get_line_points()
        poly_points = np.concatenate((np.int32(left_line_points)[::-1], np.int32(right_line_points)))
        self.left_line_object.draw_line(out_image)
        self.right_line_object.draw_line(out_image)
        cv2.fillPoly(out_image, np.int32([poly_points]), (0, 255, 0))

        # convert "bird view" to normal
        normal_image = self.transform_image(out_image, self.M2)

        # Combine the result with the original image
        result = cv2.addWeighted(img, 1, normal_image, 0.6, 0)

        return result

    @staticmethod
    def transform_image(img, m):
        '''
        WarpPerspective function wrapper
        '''
        warped = cv2.warpPerspective(
            img,
            m, (img.shape[1], img.shape[0]),
            flags=cv2.WARP_FILL_OUTLIERS + cv2.INTER_CUBIC)

        return warped

    @staticmethod
    def filter_line_colors(img):
        '''
        Filters white and yellow lines
        '''
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

    @staticmethod
    def hls_select(img, channel=0):
        '''
        Returns one of HLS channels
        '''
        hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        return hls[:, :, channel]

    @staticmethod
    def find_lane_pixels(binary_warped):
        '''
        Direct search using iteration with window
        '''
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

    def fit_polynomial(self, binary_warped):
        '''
        Tries to find lane pixels and fit them to ax^2+bx+c
        '''
        # Find our lane pixels first
        leftx, lefty, rightx, righty, out_img = self.find_lane_pixels(binary_warped)

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

    @staticmethod
    def search_around_poly(binary_warped, left_fit, right_fit):
        '''
        Try to reuse data from previous step in order to find the lines
        '''
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
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)

        ploty = np.linspace(0, binary_warped[0] - 1, binary_warped[0])

        left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
        right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

        ## Visualization ##
        # Create an image to draw on and an image to show the selection window
        out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255

        # Color in left and right line pixels
        out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

        return left_fitx, right_fitx, out_img, left_fit, right_fit

    @staticmethod
    def channel_threshold(channel, min=120, max=255):
        '''
        Channel threshold
        '''
        binary = np.zeros_like(channel)
        binary[(channel > min) & (channel <= max)] = 1

        return binary

    @staticmethod
    def erode_and_dilate(binary):
        '''
        Apply erode and dilate to remove the noise
        '''
        kernel = np.ones((3, 3), np.uint8)
        erosion = cv2.erode(binary, kernel, iterations=1)

        kernel = np.ones((12, 12), np.uint8)
        dilation = cv2.dilate(erosion, kernel, iterations=2)
        return dilation

    @staticmethod
    def binary_output_sobel(img):
        '''
        Apply sobelX and threshold
        '''
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        kernel = np.ones((3, 3), np.float32) / 25
        dst = cv2.filter2D(gray, -1, kernel)
        sobelx = cv2.Sobel(dst, cv2.CV_64F, 1, 0, ksize=5)
        abs_sobel_x = np.absolute(sobelx)

        scaled_sobel = np.uint8(255 * abs_sobel_x / np.max(abs_sobel_x))
        binary_output_sobel = np.zeros_like(scaled_sobel)
        binary_output_sobel[(scaled_sobel >= 45) & (scaled_sobel <= 120)] = 1

        return binary_output_sobel

    @staticmethod
    def plot_result(res):
        '''
        Plot result
        '''
        LineDetector.plot_images_map({"result": res[0], "undistorted_image": res[1], "transformed_image": res[2]}, columns=3,
                        img_size=(30, 50))
        LineDetector.plot_images_map({"binary_output_sobel_x": res[3], "binary_s": res[4], "dilation": res[5]}, columns=3,
                        img_size=(30, 50))
        LineDetector.plot_images_map({"masked_transformed_image": res[6], "filtered_image": res[7], "gray": res[8]}, columns=3,
                        img_size=(30, 50))
        LineDetector.plot_images_map({"binary": res[9], "out_image": res[10]}, columns=3, img_size=(30, 50))

    @staticmethod
    def plot_images_map(images, img_size=(20, 15), columns=5):
        '''
        Display image map
        '''
        plt.figure(figsize=img_size)

        i = 0
        for file_name in images:
            plt.subplot(len(images) / columns + 1, columns, i + 1).set_title(file_name)
            plt.imshow(images[file_name])
            i += 1