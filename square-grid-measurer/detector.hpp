#ifndef OPENCV_TOY_PROJECT_DETECTOR_HPP
#define OPENCV_TOY_PROJECT_DETECTOR_HPP
#include <vector>

#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>

class SquareGridDetector {
public:
    explicit SquareGridDetector(cv::Scalar low_HSV_thresh, cv::Scalar high_HSV_thresh, int min_visible_lines=5,
                                double grid_angle_tol = CV_PI/30, int min_line_votes=100, double canny_low_thresh_mul=0.66,
                                double canny_high_tresh_mul=1.33, int gauss_window_size=5, double gauss_sigma=1,
                                int morph_close_size=9);

    std::vector<cv::Point2f> detect(const cv::Mat& image);

private:
    void compute_color_mask(const cv::Mat& image);

    std::vector<cv::Point2f> grid_points(std::vector<cv::Vec3f>& lines);

    // parameters
    cv::Scalar low_HSV_thresh_;
    cv::Scalar high_HSV_thresh_;
    cv::Size gauss_window_size_;
    double gauss_sigma_;
    cv::Mat morph_close_element_;
    double canny_low_thresh_mul_;
    double canny_high_thresh_mul_;
    int min_line_votes_;
    int min_visible_lines_;
    double grid_angle_tol_;

    // intermediate results cache (to prevent always creating/resizing matrices)
    cv::Mat blurred_image_;
    cv::Mat hsv_image_;
    cv::Mat mask_;
    cv::Mat grayscale_image_;
    cv::Mat edge_mask_;
};


#endif //OPENCV_TOY_PROJECT_DETECTOR_HPP
