# Grid Detector
OpenCV grid detector. A toy project to brush up on OpenCV image processing basics, 
not meant as a very robust/efficient detector.

![Example](example.gif)

## Example usage

```c++
grid_detector::GridDetector detector; // see docstrings to set non-default params
cv::Mat image = ...;
std::vector<cv::Point2f> corners = detector.detect(image);
```