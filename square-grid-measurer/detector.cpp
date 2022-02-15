#include "detector.hpp"

#include <opencv2/imgproc.hpp>
#include <opencv2/core.hpp>

SquareGridDetector::SquareGridDetector(cv::Scalar low_HSV_thresh, cv::Scalar high_HSV_thresh, int min_visible_lines,
                                       double angle_tolerance, double line_vote_ratio_tol, int min_line_votes,
                                       double canny_low_thresh_mul, double canny_high_thresh_mul, int gauss_window_size,
                                       double gauss_sigma, int morph_close_size) :
        low_HSV_thresh_(std::move(low_HSV_thresh)), high_HSV_thresh_(std::move(high_HSV_thresh)),
        gauss_window_size_(gauss_window_size, gauss_window_size), gauss_sigma_(gauss_sigma),
        morph_close_element_(cv::getStructuringElement(cv::MORPH_RECT, cv::Size(morph_close_size, morph_close_size))),
        canny_low_thresh_mul_(canny_low_thresh_mul), canny_high_thresh_mul_(canny_high_thresh_mul),
        min_line_votes_(min_line_votes), min_visible_lines_(min_visible_lines), angle_tolerance_(angle_tolerance),
        line_vote_ratio_tol_(line_vote_ratio_tol){}

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
    auto avg_brightness = cv::mean(grayscale_image_, mask_)(0);
    cv::Canny(grayscale_image_, edge_mask_,
              avg_brightness * canny_low_thresh_mul_, avg_brightness * canny_high_thresh_mul_);
    cv::bitwise_and(edge_mask_, mask_, edge_mask_); // apply the color mask

    // TODO: We could better filter based on "unbroken" line lengths if we did probabilistic Hough transform
    // hough lines transform (single-pixel & 1° resolution)
    std::vector<cv::Vec3f> lines;
    constexpr auto one_degree_rad = CV_PI / 180;
    cv::HoughLines(edge_mask_, lines, 1, one_degree_rad, min_line_votes_);

    // filter lines & find intersections
    return grid_points(lines);
}

void SquareGridDetector::compute_color_mask(const cv::Mat& image) {
    // to HSV
    cv::cvtColor(blurred_image_, hsv_image_, cv::COLOR_BGR2HSV);
    // check HSV range -> binary mask
    cv::inRange(hsv_image_, low_HSV_thresh_, high_HSV_thresh_, mask_);
    // close holes in the mask
    cv::morphologyEx(mask_, mask_, cv::MORPH_CLOSE, morph_close_element_);
    // dilate to avoid any line detections on the edges of color mask later on
    cv::dilate(mask_, mask_, morph_close_element_);
    //TODO: could take the largest contour
}

std::vector<cv::Point2f> SquareGridDetector::grid_points(std::vector<cv::Vec3f>& lines) {
    // extract angles + a trick to mirror angles larger than 90 degrees for angle distances to make sense in euclidean space
    // TODO: this may need to be stored in a (N, 1) cv::Mat for kmeans to work (cv::Mat(line_angles) should just work, without the need to copy)
    std::vector<double> line_angles(lines.size());
    std::transform(lines.begin(), lines.end(), line_angles.begin(),
                   [](auto& line) { return line[1] <= CV_PI ? line[1] : 2 * CV_PI - line[1]; });

    // cluster into 2 groups based on line angles (assumed to find horizontal and vertical group)
    std::vector<int> clusters(line_angles.size());
    auto kmeans_criteria = cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS,
                                            15, CV_PI / 360); // 0.5 degree criteria with max. 15 iterations
    // 5 attempts with different cluster initializations
    cv::kmeans(line_angles, 2, clusters, kmeans_criteria, 5,
               cv::KmeansFlags::KMEANS_RANDOM_CENTERS);

    // TODO: perhaps check that the two groups are separated by at least some angle

    // isolate the first large enough group of lines (based on votes & angle tolerance) in the two groups

    // TODO: find representatives for each group (max votes)
    std::pair<std::vector<cv::Vec3f>, std::vector<cv::Vec3f>> line_groups;
    for (auto group=0; group < 2; ++group){
        //TODO: min_visible_lines_ + angle_tolerance_ + ratio test
    }

    // compute intersections

    return {};
}
