# Vehicle Detection and Tracking Project

## Useful information

Project goals/checkpoints:
- construct a pipeline for processing video frames;
- the pipeline must properly and accurately detect lane markings on the road, 
regardless of shadows and changes in lightning conditions;
- test the pipeline on a set of example images and detect the lane accurately
- detect and calculate lane curvature and lane circle radius
- after testing, proceed and export a video file with the above information overlaid
on top of the video stream.

## Files of interest
- images in the [examples folder](./examples) some used in this writeup, many more available
- [main.py](./main.py) - this is what is supposed to be run to fully process video

---

## Image processing Pipeline

The vehicle detection pipeline used in this project will be assessed here. (Almost) 
The same pipeline used to detect vehicles in a single image is used to detect 
vehicles in video frames as well.

### Histogram of Oriented Gradients (HOG)

- describe methods for HOG extraction, parameters and reasons for having those parameters.
- identify where is this present in the project code 

| Parameter | Chosen value |
|-----------|-------|
| Color space | HSV |
| Orientations | 12 |
| Pixels per cell | 8 |
| Cells per block | 2 |

### Sliding Window Search

- sliding window search has been implemented
- overlapping tiles are classified as vehicle/non-vehicle
- justification for the implementation is given

How did you decide what scales to search and how much to overlap windows?

Some discussion is given around how you improved the reliability of the classifier 
i.e., fewer false positives and more reliable car detections 
(this could be things like choice of feature vector, thresholding the decision 
function, hard negative mining etc.)

## Video implementation

Provide a link to your final video output. Your pipeline should perform reasonably well 
on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long 
as you are identifying the vehicles most of the time with minimal false positives.)

Describe how (and identify where in your code) you implemented some kind of filter 
for false positives and some method for combining overlapping bounding boxes.

## Discussion

Discussion includes some consideration of problems/issues faced, what could be 
improved about their algorithm/pipeline, and what hypothetical cases would cause 
their pipeline to fail.

Briefly discuss any problems / issues you faced in your implementation of this project.
Where will your pipeline likely fail? What could you do to make it more robust? 