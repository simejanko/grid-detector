#ifndef OPENCV_TOY_PROJECT_DETECTOR_HPP
#define OPENCV_TOY_PROJECT_DETECTOR_HPP

#include <vector>

#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>

class SquareGridDetector {
public:
    explicit SquareGridDetector(cv::Scalar low_HSV_thresh, cv::Scalar high_HSV_thresh, int min_visible_lines = 6,
                                double angle_tolerance = CV_PI / 45, double line_vote_ratio_tol = 0.65,
                                int min_line_votes = 150, double canny_low_thresh_mul = 0.2,
                                double canny_high_tresh_mul = 0.3, int gauss_window_size = 5, double gauss_sigma = 1,
                                int morph_close_size = 9, bool draw_debug = false, float debug_scale = 0.2);

    [[nodiscard]] std::vector<cv::Point2f> detect(const cv::Mat& image);

    [[nodiscard]] cv::Mat debug_image();

private:
    void compute_color_mask();

    std::vector<cv::Point2f> grid_points(const std::vector<cv::Vec3f>& lines);

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
    double angle_tolerance_;
    double line_vote_ratio_tol_;

    bool draw_debug_;
    float debug_scale_;

    // intermediate results cache (to prevent always creating/resizing matrices)
    cv::Mat blurred_image_;
    cv::Mat hsv_image_;
    cv::Mat mask_;
    cv::Mat grayscale_image_;
    cv::Mat edge_mask_;

    std::array<cv::Mat, 6> debug_imgs_;
};


#endif //OPENCV_TOY_PROJECT_DETECTOR_HPP
