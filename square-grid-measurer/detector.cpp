#include "detector.hpp"

#include <opencv2/imgproc.hpp>
#include <opencv2/core.hpp>

#include <cmath>
#include <list>
#include <algorithm>
#include <optional>

const int DEBUG_LINE_THICKNESS = 3;

SquareGridDetector::SquareGridDetector(cv::Scalar low_HSV_thresh, cv::Scalar high_HSV_thresh,
                                       int min_visible_lines, double angle_tolerance, double line_vote_ratio_tol,
                                       int min_line_votes, double canny_low_thresh_mul, double canny_high_thresh_mul,
                                       int gauss_window_size, double gauss_sigma, int morph_close_size, bool draw_debug,
                                       float debug_scale) :
        low_HSV_thresh_(std::move(low_HSV_thresh)), high_HSV_thresh_(std::move(high_HSV_thresh)),
        gauss_window_size_(gauss_window_size, gauss_window_size), gauss_sigma_(gauss_sigma),
        morph_close_element_(cv::getStructuringElement(cv::MORPH_RECT, cv::Size(morph_close_size, morph_close_size))),
        canny_low_thresh_mul_(canny_low_thresh_mul), canny_high_thresh_mul_(canny_high_thresh_mul),
        min_line_votes_(min_line_votes), min_visible_lines_(min_visible_lines), angle_tolerance_(angle_tolerance),
        line_vote_ratio_tol_(line_vote_ratio_tol), draw_debug_(draw_debug), debug_scale_(debug_scale) {}

/** Draws hough lines on the image. Code mostly copied from OpenCV documentation */
template<class ForwardIter>
void draw_hough_lines(ForwardIter begin, ForwardIter end, cv::Mat& image, int thickness = DEBUG_LINE_THICKNESS,
                      const cv::Scalar& color = cv::Scalar(0, 255, 0)) {
    for (auto it = begin; it != end; ++it) {
        float rho = (*it)[0], theta = (*it)[1];
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

std::vector<cv::Point2f> SquareGridDetector::detect(const cv::Mat& image) {
    if (image.depth() != CV_8U || image.channels() != 3) { //accept RGB images
        throw std::invalid_argument("Detector only accepts RGB images.");
    }

    // gauss filter
    cv::GaussianBlur(image, blurred_image_, gauss_window_size_, gauss_sigma_);
    compute_color_mask();

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

    if (draw_debug_) {
        debug_imgs_[0] = blurred_image_;
        // grayscale images are converted to BGR to match other images
        cv::cvtColor(mask_, debug_imgs_[1], cv::COLOR_GRAY2BGR);
        cv::cvtColor(edge_mask_, debug_imgs_[2], cv::COLOR_GRAY2BGR);
        for (auto i = 3; i < debug_imgs_.size(); ++i) {
            image.copyTo(debug_imgs_[i]);
        }
        draw_hough_lines(lines.cbegin(), lines.cend(), debug_imgs_[3], DEBUG_LINE_THICKNESS / debug_scale_);
    }

    // filter lines & find intersections
    return grid_points(lines);
}

void SquareGridDetector::compute_color_mask() {
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

/**
 *
 * A trick to mirror angles larger than 90 degrees for line angle distances to make sense in euclidean space
 * @param angle angle, between 0 and 180 degrees, in radians
 */
inline float angle_dist_convert(double angle) {
    return angle <= CV_PI / 2 ? angle : CV_PI - angle;
}

/**  Clusters line detections from Hough transform into 2 groups (presumably horizontal & vertical) based on angles */
std::array<std::list<cv::Vec3f>, 2> cluster_lines(const std::vector<cv::Vec3f>& lines) {
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

    std::array<std::list<cv::Vec3f>, 2> line_groups;
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
    if (!is_valid){
        return {};
    }

    return {cv::Point2f(intersection(0, 0), intersection(1, 0))};
}

std::vector<cv::Point2f> SquareGridDetector::grid_points(const std::vector<cv::Vec3f>& lines) {
    if (lines.size() < 2 * min_visible_lines_) { // too few visible lines
        return {};
    }

    auto line_groups = cluster_lines(lines);

    // filter lines
    for (auto& line_group: line_groups) {
        // line with the most votes assumed to be on the grid
        const auto& representative_line =
                *std::max_element(line_group.begin(), line_group.end(),
                                  [](const auto& line1, const auto& line2) { return line1[2] < line2[2]; });

        // remove lines whose angle is too different or votes ratio is too weak, compared to representative line
        line_group.remove_if(
                [&representative_line, rep_angle = angle_dist_convert(representative_line[1]), this](const auto& line) {
                    return std::abs(angle_dist_convert(line[1]) - rep_angle) > angle_tolerance_ ||
                           line[2] / representative_line[2] < line_vote_ratio_tol_;
                });

        // vote-based non-maxima suppression on remaining lines,
        // where 2 lines from the same group are close enough when their intersection is within an image
        for (auto it = line_group.begin(); it != line_group.end();) {
            auto any_better_lines =
                    std::any_of(line_group.cbegin(), line_group.cend(),
                                [this, l1 = *it](const auto& l2) {
                                    if (l1[2] > l2[2]) { // l1 has more votes
                                        return false;
                                    }

                                    // now check if lines are "close enough"
                                    auto inters = line_intersection(l1, l2);
                                    if (!inters) {
                                        // assumed parallel lines, compare based on rho &
                                        // very rough image-scaling heuristic threshold
                                        return std::abs(l1[0] - l2[0]) < blurred_image_.cols/100.;
                                    }
                                    auto pt = inters.value();
                                    auto within_bounds = pt.x >= 0 && pt.x < blurred_image_.cols &&
                                                         pt.y >= 0 && pt.y < blurred_image_.rows;
                                    return within_bounds;
                                });
            if (any_better_lines) {
                it = line_group.erase(it);
            }
            else {
                ++it;
            }
        }

        // not enough lines detected, return empty result
        if (line_group.size() < min_visible_lines_) {
            return {};
        }
    }

    // compute intersections
    std::vector<cv::Point2f> intersections(line_groups[0].size() * line_groups[1].size());
    for (const auto& line1: line_groups[0]) {
        for (const auto& line2: line_groups[1]) {
            auto intersection = line_intersection(line1, line2);
            if (intersection){
                intersections.push_back(intersection.value());
            }
        }
    }

    if (draw_debug_) {
        draw_hough_lines(line_groups[0].cbegin(), line_groups[0].cend(), debug_imgs_[4],
                         DEBUG_LINE_THICKNESS / debug_scale_, cv::Scalar(255, 0, 0));
        draw_hough_lines(line_groups[1].cbegin(), line_groups[1].cend(), debug_imgs_[4],
                         DEBUG_LINE_THICKNESS / debug_scale_, cv::Scalar(0, 255, 0));
        for (const auto& point: intersections) {
            cv::drawMarker(debug_imgs_[5], point, cv::Scalar(0, 255, 0), cv::MARKER_CROSS,
                           3*DEBUG_LINE_THICKNESS / debug_scale_, DEBUG_LINE_THICKNESS / debug_scale_);
        }
    }

    return intersections;
}

/**
 * Concatenates all debug images (blurred image, mask, edges, lines, filtered/grouped lines, intersections)
 * in a grid/subplot and returns resulting debug image.
 */
cv::Mat SquareGridDetector::debug_image() {
    cv::Mat first_row, second_row, result, result_scaled;
    cv::hconcat(debug_imgs_.data(), 3, first_row);
    cv::hconcat(debug_imgs_.data() + 3, 3, second_row);
    cv::vconcat(first_row, second_row, result);
    cv::resize(result, result_scaled, cv::Size(), debug_scale_, debug_scale_);
    return result_scaled;
}
