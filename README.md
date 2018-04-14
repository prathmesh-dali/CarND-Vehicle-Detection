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
[image2]: ./output_images/vehile_not_vehicle_hog.png
[image3]: ./output_images/boxes_drawn.png
[image4]: ./output_images/identified_boxes.png
[image5]: ./output_images/heat_map.png
[image6]: ./output_images/labeled.png
[image7]: ./output_images/Final_Output.png
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

I tried various combinations of parameters and colorspaces and finally I setteled on YUV colorspace with usage of `Y` channel for HOG computation since it contains more structural details as compared to other colorspaces and other channels of `YUV` color space. I tried with other colorspaces as well like `HSV` with `H` and `V` channel for computation of `HOG`, but got better and accurate results for `Y` channel of `YUV` colorspace. For `orientation` and `pixels_per_cell` selection I tried with differnt combination of both such as `11, (8, 8), 4, (8, 8),` etc resp. And then I setteld on orientation `7` and pixels_per_cell `(16, 16)` as tradeof between computation time for test image and accuracy.

Below table list down some of the trials training accuracy and testing time:-

| Kernel | Color Space| pixels_per_cell| Orientation| HOG channel | Accuracy | Training Time| Testing Time|
| --- | --- | --- | --- | --- | --- | --- |  --- |
|Linear|HSV|(8, 8)|7|H, V | 97.83 | 18 | 5 |
|rbf|HSV|(8, 8)|7|H, V | 98.2 | 60 | 6 |
|rbf|YUV|(8, 8)|7|Y | 98.7 | 63 | 1.32 |
|rbf|YUV|(16, 16)|7|Y | 99.07 | 64 | 1.24 |

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

For training of `SVM` I used Image binning features, color histogram features and `HOG` features. For color histogram I used `32 bins`. For Image binning the image is resized to `32 X 32 ` before extracting raw image pixels. For `HOG` computation I used `Y` channel of `YUV` colorspace. I trained a `RBF kernel SVM` using color histogram features, raw pixel features and `Y` channel `HOG`. Total number of features I used for training `SVM` are `1116` for each `64x64` image. It is trained in `64` seconds and got accuracy of `99.07%`. I tried by using `Linear kernel SVM` with same dataset. I was getting accuracy of `96.17%` with training time of `18` seconds. But while running sliding window on test image I was getting lots of false positive. Even I compared for video processing time at the end and I have not found huge difference between processing time of both, so for better accuracy I opted for `RBF kernel SVM`.

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

For sliding window search I am using 3 window size `(64, 64), (80, 80) and (96, 128)`. For `(64, 64)` I am searching from `400 pixel` to `500 pixe`l. For `(80, 80)` I am searching from `400 pixel` to `550 pixel`. For `(96, 128)` I am searching from `400 pixel` to `600 pixel`.

For each window I have selected region such that it can contain atleast `3` rows of window search. For selecting overlap I tried with multiple values. Finally I setteled with `0.5` for time accuracy tradeoff. 

Below image shows all the windows I have used on region of image.
![alt text][image3]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on three scales using `YUV Y-channel HOG` features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.

To compensate `wobble effect` I am taking average of last 10 frame's heatmap to plot boxes on image. 

Here are some example images:

![alt text][image4]
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video_output.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a test images, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are eight images and their corresponding heatmaps:

![alt text][image5]

### Here is the output of `scipy.ndimage.measurements.label()` on the heatmap for fourth test image:
![alt text][image6]

### Here the resulting bounding boxes are drawn onto the fourth image:
![alt text][image7]



---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

- First and foremost challenge I faced which features I should include which are not. For this I checked performance of both `Linear` and `RBF` kernel on different colorspaces, pixel_per_cell, histogram bin sizes, orientations and spatial_size. And finally setteled on `YUV` colorspace binnig features, `YUV` colorspace color histogram features, `YUV` colorspace `Y` channel HOG features, pixel_per_cell 16, histogram bin 32, orientation 7 and spatial_size 16 X 16. Where I was getting better accuracy for both the kernels compared to time taken by the kernel to train and output test image. This process took my a day of efforts to settel down on values.
- Then the problem is that which kernel to use. I started with `Linear kernel of SVM` but I was getting lots of false positives and even at some places the car was not detecting even though I had accuracy of arround 98%. I tried varying values of `C` and `gamma` without any success. Then I switched to `RBF kernel` where I was getting better results than its counterpart.  I spent hours while figuring out `kernel`.
- Now next callenge is to select window size and region to run those windows. For this I tried differnt combination of windows of different shapes and finally setteled on my better working combination by considering time and acuracy tradeoff.
- Now the time to remove false positive and wobble effect. First of all to remove false positives I have taken threshold on heatmap. I tried with different values of threshold and settled on value 1 since for higher values of threshod I was loosing vehicle in some of the frames and for lower values of threshold I was getting more false positives. Then I have taken average of last 10 frames heat map to generate heatmap of current frame to counter wobble effect which inturn helped in removing some of the false positives. Then I placed restriction on boxes to be drawn to counter false positives.

- The pipeline can fail in following scenarios:
    - The vehicle is smaller than 30 X 30 pixels in image.
    - To identify Non car vehicle.
    - In bad lightning conditions.

- Below are the points which can be improved to make pipeline robust:
    -  There are few places where I am getting false positives in my video those can be removed by optimizing features used for  training model with better configurations of hyper parameters.
    - We can see in video even vehicles at opposite side of the road are getting detected. This can trick our vehicle so we can apply some kind of filter so that vehicls on opposite side of road cann't be detected.
    - since I am taking average of last 10 frames we can see there is no wobble effect but sometimes bouding boxes are lagging behind. So that can be improved by reducing the number frames considered for taking average.
    - Video processing takes 20 minutes to process 50 seconds video. That can be reduced by reducing number of features and by using `linear` kernel of `SVM`. 

