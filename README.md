**Advanced Lane Finding Project**

The goals/steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to the center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[distorted_img]: ./output_images/distorted.jpg "Distorted"
[undistorted_img]: ./output_images/undistorted.jpg "Undistorted"
[raw_image]: ./test_images/test3.jpg "Raw image"
[undistorted_image]: ./output_images/undistorted_image.jpg "Undistorted image"
[transformed_image]: ./output_images/transformed_image.jpg "Bird view"
[binary_output_sobel_x]: ./output_images/binary_output_sobel_x.jpg "Binary Sobel X"
[binary_s]: ./output_images/binary_s.jpg "Binary S channel"
[dilation]: ./output_images/dilation.jpg "Erode and Dilate"
[masked_transformed_image]: ./output_images/masked_transformed_image.jpg "Masked transformed image"
[filtered_image]: ./output_images/filtered_image.jpg "Color filtered image"
[gray]: ./output_images/gray.jpg "Gray"
[binary]: ./output_images/binary.jpg "Binary"
[out_image]: ./output_images/out_image.jpg "Output image"
[result]: ./output_images/result.jpg "Result"
[bird_view_straight_lines]: ./output_images/bird_view_straight_lines.jpg "Result"
[project_video]: ./output_images/project_video.mp4 "Project video"


### Camera Calibration

#### 1. Example

| Distorted  | Undistorted  |
|---|---|
| ![alt text][distorted_img]  | ![alt text][undistorted_img]  |

#### 2. Description

The code for this step is in `calibration.py` file. Which defines global vars `ret, mtx, dist, rvecs, tvecs` as result of `cv2.calibrateCamera` function. Images for calibration are stored in `camera_cal` folder. 

Some of the calibration images have a different amount of visible corners and calibration code takes it into account. In case if we will exclude those images, the calibration result is worth.

### Pipeline (single images)

The whole pipeline is defined within `pipeline.py` in the object called `LineDetector`.

#### 1. Undistort image.

At this step using camera calibration results and `cv2.undistort` function, we are acheiving the folowing image from the original one:

| Raw image  | Undistorted image  |
|---|---|
| ![alt text][raw_image]  | ![alt text][undistorted_image]  |

#### 2. Create a "bird view"

One of the most important steps in the whole pipeline is creating a bird view. Parameters for the transformation can be found manually or calculated. 
In my case after trying different combinations, I choose the following:

```python
src = np.float32([[580, 460], [205, 720], [1110, 720], [703, 460]])
dst = np.float32([[320, 0], [320, 720], [960, 720], [960, 0]])  
```

| Undistorted image  | Bird view |
|---|---|
| ![alt text][undistorted_image]  | ![alt text][transformed_image]  |

#### 3. Apply Sobel operator to "bird view" in the x-direction

The current step is about to apply Sobel operator on "bird view", take the absolute value of result and use the threshold (found manually) to get a binary mask.

| Bird view  | Absolute Sobel X |
|---|---|
| ![alt text][transformed_image]  | ![alt text][binary_output_sobel_x]  |

Test on parallel lines:

![alt text][bird_view_straight_lines]

#### 4. Threshold S channel of "bird view" to generate a binary mask

At this step, the goal is to extract useful info from the image channels. For that, I converted the image to HLS and used the S channel with the threshold.

| Bird view  | Thresholded S channel |
|---|---|
| ![alt text][transformed_image]  | ![alt text][binary_s]  |

#### 5. Erode and dilate "thresholded S channel mask" and "absolute Sobel x mask" 

At this point, we are combining (operator `OR`) "thresholded S channel mask" and "absolute Sobel x mask". 
And to remove noise apply sequentially `cv2.erode` and `cv2.dilate` functions.

| Absolute Sobel X | Thresholded S channel | Erode and dilate |
|---|---|---|
| ![alt text][binary_output_sobel_x]  | ![alt text][binary_s]  | ![alt text][dilate]  |

#### 7. Mask transformed image

Here apply the mask from the previous step to the "bird view".

| Bird view | Erode and dilate | Masked bird view |
|---|---|---|
| ![alt text][transformed_image]  | ![alt text][dilate]  | ![alt text][masked_transformed_image]  |


#### 8. Using the color filter to find lines

Convert image to HSV and use color information to find lines on the masked "bird view".

| Masked bird view | Filtered image |
|---|---|
| ![alt text][masked_transformed_image]  | ![alt text][filtered_image]  |



#### 9. Create a binary image from the color-filtered images

Color filtered image to gray and then to binary

| Filtered image | Gray | Binary |
|---|---|---|
| ![alt text][filtered_image]  | ![alt text][gray]  | ![alt text][binary] |


#### 10. Create a binary image from the color-filtered images

At this step, I try to find a polynomial approximation (of degree 2).

- First I try using the histogram to find the origin of line and then using the sliding window, step by step discover all the point.
Implemented in `LineDetector.fit_polynomial`.

- In case if lines were detected at the previous step, I try to reuse this info assuming that the new polynomial should have similar params.
Implemented in `LineDetector.search_around_poly`.


| Binary | Polynomial |
|---|---|
| ![alt text][binary]  | ![alt text][out_image]  |

#### 11. Draw lines, calculate curvature, distance to the line, draw lines, poly, etc.

Distance and curvature calculation is in `line.py` class Line, methods `measure_curvature_real` and `measure_distance_real` 

| Raw | Result |
|---|---|
| ![alt text][raw_image]  | ![alt text][result]  |

---

### Pipeline (video)

#### 1. Here is a result of using pipeline on the project video file:

Here's a [link to my video result](./output_images/project_video.mp4)

---

### Discussion and conclusion

#### 1. What is good about this pipeline?

I found this pipeline easy to understand and enough robust for the good conditions like: 

- highway 
- good weather 
- daylight 
- lines are visible
- the road surface is in good condition: no cracks or stains 

#### 2. When it is not good enough? 

Cases when it will fail, for instance:

- Not stable lighting condition: when a car is driving on the highway and then in the shade under the bridge or in the forest when the sun is blinking because of the trees.
- Yellow or white things, not lines, are on the street, big enough that filter is not filtering them.
- When road can't be approximated with a polynomial of 2 degrees (very curved road).
- When the yellow or white car will drive close enough.
- etc.

 #### 3. How to improve?
 
 - Add `memory` to the line that it will average the polynomial params for the last "n" detection. 
 This will provide "smoother" lines change and will provide direction if for whatever reason lines can't be detected.
 
 - Add a `reasonable` threshold for polynomial params, the road is a subject from real worlds. its curvature can't change for billion times in nanosecond. So we can skip outliers.
 
 - Use a better camera, quicker brightness adjustments and high-resolution image could help.
 
 - Use `dynamic` threshold parameters adjustment, current pipeline depends on the lighting, mostly because of hardcoded threshold parameters. 
 It will be great to use a different strategy of line detection for different situation, use error detection or just by looking at the average image brightness.
