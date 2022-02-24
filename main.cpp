#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>

#include "grid-detector/detector.hpp"

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Requires one argument: path to video or OpenCV camera id" << std::endl;
        return 1;
    }
    std::string video_arg = argv[1];

    cv::VideoCapture cap;
    // try to interpret as OpenCV camera id, video path otherwise
    try {
        int camera_id = std::stoi(video_arg);
        cap.open(camera_id);
    } catch (std::invalid_argument& e) {
        cap.open(video_arg);
    }
    if (!cap.isOpened()) {
        std::cerr << "Unable to open video/camera!" << std::endl;
        return 2;
    }

    SquareGridDetector detector(cv::Scalar_(0, 0, 60), cv::Scalar(179, 140, 255), 2, CV_PI / 10, 150, 0.5, 1.0, 1, 5, 1,
                                3, true);
    cv::Mat frame;
    while (cap.read(frame)) {
        if (frame.empty()) {
            break;
        }

        auto corners = detector.detect(frame);
        auto debug_img = detector.debug_image();

        cv::imshow("Debug image", debug_img);
        cv::waitKey(5);
    }

    return 0;
}
