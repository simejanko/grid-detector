#include "detector.hpp"

#include <opencv2/imgproc.hpp>

SquareGridDetector::SquareGridDetector(cv::Scalar low_HSV_thresh, cv::Scalar high_HSV_thresh,
                                       double canny_low_thresh_mul, double canny_high_thresh_mul,
                                       int gauss_window_size, double gauss_sigma, int morph_close_size) :
        low_HSV_thresh_(std::move(low_HSV_thresh)), high_HSV_thresh_(std::move(high_HSV_thresh)),
        gauss_window_size_(gauss_window_size, gauss_window_size), gauss_sigma_(gauss_sigma),
        morph_close_element_(cv::getStructuringElement(cv::MORPH_RECT, cv::Size(morph_close_size, morph_close_size))),
        canny_low_thresh_mul_(canny_low_thresh_mul), canny_high_thresh_mul_(canny_high_thresh_mul) {}

std::vector<cv::Point2f> SquareGridDetector::detect(const cv::Mat& image) {
    if (image.depth() != CV_8U || image.channels() != 3) { //accept RGB images
        throw std::invalid_argument("Detector only accepts RGB images.");
    }

    // gauss filter
    cv::GaussianBlur(image, blurred_image_, gauss_window_size_, gauss_sigma_);

    compute_color_mask(image);

    // canny edges within a mask
    cv::cvtColor(blurred_image_, grayscale_image_, cv::COLOR_BGR2GRAY);
    // set canny thresholds relative to average brightness in masked area - very rough heuristic
    double avg_brightness = cv::mean(grayscale_image_, mask_)(0);
    cv::Canny(grayscale_image_, edge_mask_,
              avg_brightness * canny_low_thresh_mul_, avg_brightness * canny_high_thresh_mul_);
    cv::bitwise_and(edge_mask_, mask_, edge_mask_); // apply the color mask

    // hough transform

    // nms magic (e.g. find the best big enough cluster of lines)

    // intersections
    return std::vector<cv::Point2f>();
}

void SquareGridDetector::compute_color_mask(const cv::Mat& image) {
    // to HSV
    cv::cvtColor(blurred_image_, hsv_image_, cv::COLOR_BGR2HSV);
    // check HSV range -> binary mask
    cv::inRange(hsv_image_, low_HSV_thresh_, high_HSV_thresh_, mask_);
    // close holes in the mask
    cv::morphologyEx(mask_, mask_, cv::MORPH_CLOSE, morph_close_element_);
}
