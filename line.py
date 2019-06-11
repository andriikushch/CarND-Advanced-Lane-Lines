import numpy as np
import cv2


class Line:
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False
        # polynomial coefficients for the most recent fit
        self.current_fit = np.array([False, False, False])
        # x values for detected line pixels
        self.allx = []
        # y values for detected line pixels
        self.ally = []
        # meters per pixel in x dimension
        self.xm_per_pix = 3.7 / 700
        # meters per pixel in y dimension
        self.ym_per_pix = 1 / 30

    def get_line_points(self):
        '''
        Returns line's points
        '''
        points = np.dstack((self.allx, self.ally))[0]

        return points

    def draw_line(self, img):
        '''
        Draws line on given image
        '''
        cv2.polylines(img, np.int32([self.get_line_points()]), False, (255, 150, 0), 1)

    def measure_curvature_pixels(self, image):
        '''
        Calculates the curvature of polynomial functions in pixels.
        '''
        # Define y-value where we want radius of curvature
        # We'll choose the maximum y-value, corresponding to the bottom of the image
        y_eval = image.shape[1]

        # Calculation of R_curve (radius of curvature)
        curverad = ((1 + (2 * self.current_fit[0] * y_eval + self.current_fit[1]) ** 2) ** 1.5) / np.absolute(
            2 * self.current_fit[0])

        return curverad

    def measure_curvature_real(self, image):
        '''
        Calculates the curvature of polynomial functions in meters.
        '''
        y_eval = image.shape[1]

        polfit_real = np.polyfit(self.ally * self.ym_per_pix, self.allx * self.xm_per_pix, 2)

        # Calculation of R_curve (radius of curvature)
        curverad = ((1 + (
                    2 * polfit_real[0] * y_eval * self.ym_per_pix + polfit_real[1]) ** 2) ** 1.5) / np.absolute(
            2 * polfit_real[0])

        return curverad

    def measure_distance_real(self, image):
        '''
        Calculates distance to the line
        '''
        f = np.poly1d(self.current_fit)
        middle = image.shape[1] / 2

        return abs(middle - f(image.shape[1])) * self.xm_per_pix
