**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/vehile_not_vehicle.png
[image2]: ./examples/HOG_example.jpg
[image3]: ./examples/sliding_windows.jpg
[image4]: ./examples/sliding_window.jpg
[image5]: ./examples/bboxes_and_heat.png
[image6]: ./examples/labels_map.png
[image7]: ./examples/output_bboxes.png
[video1]: ./project_video.mp4

---
### Writeup / README

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YUV` color space and HOG parameters of `orientations=7`, `pixels_per_cell=(16, 16)` and `cells_per_block=(2, 2)`:


![alt text][image2]

#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and colorspaces and finally I setteled on YUV colorspace with usage of Y channel for HOG computation since it contains more structural details as compared to other colorspaces and other channels of YUV color space. I tried with other colorspaces as well like HSV with S and V channel for computation of HOG, but got better and accurate results for Y channel of YUV colorspace. For orientation and pixels_per_cell selection I tried with differnt combination of both such as 11, (8, 8), 4, (8, 8), etc. And then I setteld on orientation 7 and pixels_per_cell (16, 16) as tradeof between computation time for test image and accuracy.

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

For training of SVM I used Image binning features, color histogram features and HOG features. For color histogram I used 32 bins. For Image binning the image is resized to 32 X 32 before extracting raw image pixels. For HOG computation I used Y channel of YUV colorspace. I trained a RBF SVM using color histogram features, raw pixel features and Y channel HOG. Total number of features I used for training SVM are 1116 for each 64x64 image. It is trained in 53 seconds and got accuracy of 99.12%. I tried by using Linear SVM with same dataset. I was getting accuracy of 96.17% with training time of 18 seconds. But while running sliding window on test image I was getting lots of false positive. Even I compared for video processing time at the end and I have not found huge difference between processing time of both, so for better accuracy I opted for RBF SVM.

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

For sliding window search I am using 3 window size (64, 64), (96, 96) and (128, 128). For (64, 64) I am searching from 400 pixel to 500 pixel. For (96, 96) I am searching from 400 pixel to 550 pixel. For (128, 128) I am searching from 400 pixel to 550 pixel.

For each window I am selecting region such that it can contain atleast 3 rows of window search. For selecting overlap I tried with multiple values. Finally I seatteled with 0.5 for time accuracy tradeoff. 

While scanning horizontally I am taking start point as 500 to remove false positives at left side of image.

To compensate wobble effect I am taking average of last 10 frames heatmap to plot boxes on image. 
![alt text][image3]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on three scales using YUV Y-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image4]
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video_output.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

![alt text][image5]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image6]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]



---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

