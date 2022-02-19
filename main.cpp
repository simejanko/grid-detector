#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

#include "square-grid-measurer/detector.hpp"

int main() {
    SquareGridDetector detector(cv::Scalar_(13, 70, 55), cv::Scalar(23, 200, 150));
    cv::Mat img = cv::imread("grid2.jpg", cv::IMREAD_COLOR);
    auto corners = detector.detect(img);

    return 0;
}
