#ifndef GRID_DETECTOR_HPP
#define GRID_DETECTOR_HPP

#include <vector>

#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>

namespace grid_detector {
/**
 * Grid pattern detector for BGR images. Specifically outputs grid line intersection points on a detected grid.
 */
    class GridDetector {
    public:
        /**
         * @param low_HSV_thresh min. HSV values used for color thresholding (defaults to lowest possible HSV values)
         * @param high_HSV_thresh max. HSV values used for color thresholding (defaults to highest possible HSV values)
         * @param min_visible_lines minimum number of detected vertical/horizontal lines to successfully detect a grid
         * @param angle_tolerance angle tolerance used to filter vertical/horizontal that are too far away from the median angle in a group
         * @param min_line_votes minimum number votes per line, used when obtaining lines with Hough transform
         * @param canny_low_thresh_mul low threshold for Canny edge detector, relative to average brightness in histogram equalized image
         * @param canny_high_tresh_mul high threshold for Canny edge detector, relative to average brightness in histogram equalized image
         * @param nms_strength non-maxima suppression strength for intersection-based NMS. Defaults to 1, which means any
         *   lines within the same group (vertical/horizontal) are filtered (based on Hough transform votes), if they
         *   intersect within the image bounds. Higher values expand that range beyond image bounds (image bounds multiplier).
         *   Note: parallel lines are filtered based on distance between them instead.
         * @param gauss_window_size window size for gauss filtering of the initial image
         * @param gauss_sigma sigma for gauss filtering of the initial image
         * @param morph_close_size filter size for morphological close operations to fill holes in a color mask
         * @param draw_debug enable drawing of debug image, consisting of 6 subimages:
         *   blurred image, color mask, edges, lines, filtered/grouped lines, intersections)
         * @param debug_scale relative scale of the debug image (debug images can get big as they consisnt of 6 subimages)
         */
        explicit GridDetector(cv::Scalar low_HSV_thresh = cv::Scalar(0, 0, 0),
                              cv::Scalar high_HSV_thresh = cv::Scalar(179, 255, 255), int min_visible_lines = 2,
                              double angle_tolerance = CV_PI / 20, int min_line_votes = 150,
                              double canny_low_thresh_mul = 0.5, double canny_high_tresh_mul = 1.0,
                              int nms_strength = 1, int gauss_window_size = 5, double gauss_sigma = 1,
                              int morph_close_size = 5, bool draw_debug = false, float debug_scale = 0.5);

        /**
         * Detects grid pattern in an image
         * @param image 8-chanel BGR images in which to detect grid pattern
         * @return grid line intersection points on a detected grid
         */
        [[nodiscard]] std::vector<cv::Point2f> detect(const cv::Mat& image);

        /**
         * Concatenates all debug images (blurred image, color mask, edges, lines, filtered/grouped lines, intersections)
         * in a grid/subplot and returns resulting debug image. Also adds titles to each subimage.
         * @throws std::runtime_error if debug image drawing wasn't enabled in constructor
         */
        [[nodiscard]] cv::Mat debug_image();

    private:
        /** Computes binary color mask for blurred image, based on HSV range */
        void compute_color_mask();

        /**
         * Filters a group of lines (vertical/horizontal) based on angle deviation from median angle and
         * intersection-based non-maxima suppression.
         */
        void filter_line_group(std::vector<cv::Vec3f>& line_group) const;

        /**
         * Computes grid line intersections, given raw Hough line detections.
         * Clusters lines into two groups (horizontal/vertical), filters them and computes pairwise intersection
         * between the line groups.
         * @return intersection points between the horizontal/vertical lines
         */
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
        int nms_strength_;

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
}

#endif //GRID_DETECTOR_HPP
