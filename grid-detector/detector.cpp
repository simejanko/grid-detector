#include "detector.hpp"

#include <cmath>
#include <list>
#include <algorithm>
#include <optional>

#include <opencv2/imgproc.hpp>
#include <opencv2/core.hpp>

const int DEBUG_LINE_THICKNESS = 3;
const std::string DEBUG_IMAGE_TITLES[] = {"Blurred image", "Color mask", "Edges", "Lines", "Clustered & filtered lines",
                                          "Intersections"};

/** Draws hough lines on the image */
void draw_hough_lines(const std::vector<cv::Vec3f>& lines, cv::Mat& image, int thickness = DEBUG_LINE_THICKNESS,
                      const cv::Scalar& color = cv::Scalar(0, 255, 0)) {
    for (auto& line : lines) {
        float rho = line[0], theta = line[1];
        cv::Point pt1, pt2;
        double a = std::cos(theta), b = std::sin(theta);
        double x0 = a * rho, y0 = b * rho;
        pt1.x = cvRound(x0 + 1e5 * (-b));
        pt1.y = cvRound(y0 + 1e5 * (a));
        pt2.x = cvRound(x0 - 1e5 * (-b));
        pt2.y = cvRound(y0 - 1e5 * (a));
        cv::line(image, pt1, pt2, color, thickness, cv::LINE_AA);
    }
}

GridDetector::GridDetector(cv::Scalar low_HSV_thresh, cv::Scalar high_HSV_thresh,
                           int min_visible_lines, double angle_tolerance,
                           int min_line_votes, double canny_low_thresh_mul, double canny_high_thresh_mul,
                           int nms_strength, int gauss_window_size, double gauss_sigma,
                           int morph_close_size, bool draw_debug, float debug_scale) :
        low_HSV_thresh_(std::move(low_HSV_thresh)), high_HSV_thresh_(std::move(high_HSV_thresh)),
        gauss_window_size_(gauss_window_size, gauss_window_size), gauss_sigma_(gauss_sigma),
        morph_close_element_(cv::getStructuringElement(cv::MORPH_RECT, cv::Size(morph_close_size, morph_close_size))),
        canny_low_thresh_mul_(canny_low_thresh_mul), canny_high_thresh_mul_(canny_high_thresh_mul),
        min_line_votes_(min_line_votes), min_visible_lines_(min_visible_lines), angle_tolerance_(angle_tolerance),
        draw_debug_(draw_debug), debug_scale_(debug_scale), nms_strength_(nms_strength) {}

std::vector<cv::Point2f> GridDetector::detect(const cv::Mat& image) {
    if (image.depth() != CV_8U || image.channels() != 3) {
        throw std::invalid_argument("Detector only accepts BGR images.");
    }

    // gauss filter
    cv::GaussianBlur(image, blurred_image_, gauss_window_size_, gauss_sigma_);
    compute_color_mask();

    // convert to grayscale and equalize histogram
    cv::cvtColor(blurred_image_, grayscale_image_, cv::COLOR_BGR2GRAY);
    equalizeHist(grayscale_image_, grayscale_image_);

    // canny edges within a mask
    // set canny thresholds relative to average brightness in masked area - very rough heuristic
    auto avg_brightness = cv::mean(grayscale_image_, mask_)(0);
    cv::Canny(grayscale_image_, edge_mask_,
              avg_brightness * canny_low_thresh_mul_, avg_brightness * canny_high_thresh_mul_);
    cv::bitwise_and(edge_mask_, mask_, edge_mask_); // apply the color mask

    // hough lines transform (single-pixel & 1° resolution)
    std::vector<cv::Vec3f> lines;
    constexpr auto one_degree_rad = CV_PI / 180;
    cv::HoughLines(edge_mask_, lines, 1, one_degree_rad, min_line_votes_);

    if (draw_debug_) {
        debug_imgs_[0] = blurred_image_;
        // grayscale debug images are converted to BGR to match other debug images
        cv::cvtColor(mask_, debug_imgs_[1], cv::COLOR_GRAY2BGR);
        cv::cvtColor(edge_mask_, debug_imgs_[2], cv::COLOR_GRAY2BGR);
        for (auto i = 3; i < debug_imgs_.size(); ++i) {
            image.copyTo(debug_imgs_[i]);
        }
        draw_hough_lines(lines, debug_imgs_[3], DEBUG_LINE_THICKNESS / debug_scale_);
    }

    // filter lines & find intersections
    return grid_points(lines);
}

void GridDetector::compute_color_mask() {
    // to HSV
    cv::cvtColor(blurred_image_, hsv_image_, cv::COLOR_BGR2HSV);
    // check HSV range -> binary mask
    cv::inRange(hsv_image_, low_HSV_thresh_, high_HSV_thresh_, mask_);
    // close holes in the mask
    cv::morphologyEx(mask_, mask_, cv::MORPH_CLOSE, morph_close_element_);
    // erode to avoid any line detections on the edges of color mask later on
    cv::erode(mask_, mask_, morph_close_element_);
}

/**
 * A trick to mirror angles larger than 90 degrees for line angle distances to make sense in euclidean space
 * @param angle angle, between 0 and PI, in radians
 */
inline float angle_dist_convert(double angle) {
    return angle <= CV_PI / 2 ? angle : CV_PI - angle;
}

/**  Clusters line detections from Hough transform into 2 groups (presumably horizontal & vertical) based on angles */
std::array<std::vector<cv::Vec3f>, 2> cluster_lines(const std::vector<cv::Vec3f>& lines) {
    // extract angles
    std::vector<float> line_angles(lines.size());
    std::transform(lines.begin(), lines.end(), line_angles.begin(),
                   [](auto& line) { return angle_dist_convert(line[1]); });
    cv::Mat line_angles_mat(line_angles);

    // cluster into 2 groups based on line angles (assumed to find horizontal and vertical group)
    std::vector<int> clusters(line_angles.size());
    auto kmeans_criteria = cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS,
                                            15, CV_PI / 360); // 0.5 degree criteria with max. 15 iterations
    // 5 attempts with different cluster initializations
    cv::kmeans(line_angles_mat, 2, clusters, kmeans_criteria, 5,
               cv::KmeansFlags::KMEANS_RANDOM_CENTERS);

    // TODO: perhaps check that the two groups are separated by at least some angle
    //  (at least we want to avoid degenerate cases where all angles are equal)

    std::array<std::vector<cv::Vec3f>, 2> line_groups;
    for (int i = 0; i < lines.size(); ++i) {
        line_groups[clusters[i]].push_back(lines[i]);
    }
    return line_groups;
}

/**
 * Computes intersection of two lines from Hough transform by solving liner system (not the most efficient way).
 * Returns empty optional if no intersection could be computed (e.g. parallel lines).
 */
std::optional<cv::Point2f> line_intersection(const cv::Vec3f& line1, const cv::Vec3f& line2) {
    auto rho1 = line1[0], theta1 = line1[1];
    auto rho2 = line2[0], theta2 = line2[1];
    cv::Mat m({2, 2}, {std::cos(theta1), std::sin(theta1), std::cos(theta2), std::sin(theta2)});
    cv::Mat b({2, 1}, {rho1, rho2});
    cv::Mat_<float> intersection;
    auto is_valid = cv::solve(m, b, intersection);
    if (!is_valid) {
        return {};
    }

    return {cv::Point2f(intersection(0, 0), intersection(1, 0))};
}

void GridDetector::filter_line_group(std::vector<cv::Vec3f>& line_group) {
    // bounds for performing non-maxima suppression on line_group
    auto nms_intersection_bounds = cv::Rect(
            cv::Point2i(-(nms_strength_ - 1) * blurred_image_.cols, -(nms_strength_ - 1) * blurred_image_.rows),
            cv::Point2i(nms_strength_ * blurred_image_.cols, nms_strength_ * blurred_image_.rows));

    // median angle in the group (approximate for even-sized groups)
    auto median_offset = line_group.size() / 2;
    std::nth_element(line_group.begin(), line_group.begin() + median_offset, line_group.end(),
                     [](const auto& line1, const auto& line2) {
                         return angle_dist_convert(line1[1]) <
                                angle_dist_convert(line2[1]);
                     });
    auto median_angle = angle_dist_convert(line_group[median_offset][1]);

    // remove line_group whose angle is too different from median
    line_group.erase(
            std::remove_if(line_group.begin(), line_group.end(),
                           [median_angle, this](const auto& line) {
                               return std::abs(angle_dist_convert(line[1]) - median_angle) > angle_tolerance_;
                           }),
            line_group.end());

    // vote-based non-maxima suppression on remaining line_group (not efficient),
    // where 2 line_group from the same group are close enough when their intersection is
    // within a multiplier of image bounds (based on nms_strength parameter)
    for (auto it = line_group.begin(); it != line_group.end();) {
        auto any_better_lines =
                std::any_of(line_group.cbegin(), line_group.cend(),
                            [this, l1 = *it, &nms_intersection_bounds](const auto& l2) {
                                // if l1 has more votes or same line_group l2 is definitely not better
                                if (l1[2] > l2[2] || l1 == l2) {
                                    return false;
                                }

                                // now check if line_group are "close enough" to eliminate one
                                auto inters = line_intersection(l1, l2);
                                if (!inters) {
                                    // assume parallel line_group, compare based on rho &
                                    // very rough size-scaling heuristic threshold
                                    return std::abs(l1[0] - l2[0]) < blurred_image_.cols / 100.;
                                }
                                auto intersection_point = inters.value();
                                return nms_intersection_bounds.contains(intersection_point);
                            });
        if (any_better_lines) {
            it = line_group.erase(it);
        }
        else {
            ++it;
        }
    }
}

std::vector<cv::Point2f> GridDetector::grid_points(const std::vector<cv::Vec3f>& lines) {
    if (lines.size() < 2 * min_visible_lines_) { // too few visible lines
        return {};
    }
    auto line_groups = cluster_lines(lines);

    // filter line groups
    for (auto& line_group: line_groups) {
        filter_line_group(line_group);
        if (line_group.size() < min_visible_lines_) { // not enough lines remaining after filtering
            return {};
        }
    }

    // compute pairwise line intersections between the line groups
    std::vector<cv::Point2f> intersections(line_groups[0].size() * line_groups[1].size());
    for (const auto& line1: line_groups[0]) {
        for (const auto& line2: line_groups[1]) {
            auto intersection = line_intersection(line1, line2);
            if (intersection) {
                intersections.push_back(intersection.value());
            }
        }
    }

    if (draw_debug_) {
        // sort line groups according to angle of first line (to make debug colors flicker less)
        auto& line_group0 = line_groups.front(), line_group1 = line_groups.back();
        if (angle_dist_convert(line_group0.front()[1]) > angle_dist_convert(line_group1.front()[1])) {
            std::swap(line_group0, line_group1);
        }

        draw_hough_lines(line_group0, debug_imgs_[4], DEBUG_LINE_THICKNESS / debug_scale_,
                         cv::Scalar(255, 0, 0));
        draw_hough_lines(line_group1, debug_imgs_[4], DEBUG_LINE_THICKNESS / debug_scale_,
                         cv::Scalar(0, 255, 0));

        for (const auto& point: intersections) {
            cv::drawMarker(debug_imgs_[5], point, cv::Scalar(0, 255, 0), cv::MARKER_CROSS,
                           3 * DEBUG_LINE_THICKNESS / debug_scale_, DEBUG_LINE_THICKNESS / debug_scale_);
        }
    }

    return intersections;
}

cv::Mat GridDetector::debug_image() {
    if (!draw_debug_){
        throw std::runtime_error("Debug image drawing is not enabled. Enable it in constructor.");
    }

    for (int i=0; i<debug_imgs_.size(); ++i){
        auto& img = debug_imgs_[i];
        cv::putText(img, DEBUG_IMAGE_TITLES[i], {img.cols / 10, img.rows / 10}, cv::FONT_HERSHEY_SIMPLEX, 1,
                    {0, 0, 255}, DEBUG_LINE_THICKNESS);
    }

    cv::Mat first_row, second_row, result, result_scaled;
    cv::hconcat(debug_imgs_.data(), 3, first_row);
    cv::hconcat(debug_imgs_.data() + 3, 3, second_row);
    cv::vconcat(first_row, second_row, result);
    cv::resize(result, result_scaled, cv::Size(), debug_scale_, debug_scale_);
    return result_scaled;
}
