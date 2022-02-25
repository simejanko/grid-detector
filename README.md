# Grid Detector
OpenCV grid detector. Not meant as a very robust/efficient detector, but used as a toy project to brush up on OpenCV image processing basics.

![Example](example.gif)

## Example usage

```c++
grid_detector::GridDetector detector(); // see docstrings to set non-default params
cv::Mat image = ...;
std::vector<cv::Point2f> corners = detector.detect(image);
```