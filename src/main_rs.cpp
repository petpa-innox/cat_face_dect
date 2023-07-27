#include <cstdlib>
#include <librealsense2/rs.hpp> // Include RealSense Cross Platform API
#include <opencv2/core/hal/interface.h>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>   // Include OpenCV API
#include <vector>


const char* const clasifier_path = "/home/holmes/code/cat_face_detection/haarcascade_frontalcatface.xml";


int main(int argc, char * argv[]) try
{
    cv::CascadeClassifier  classifier(clasifier_path);
    // Declare depth colorizer for pretty visualization of depth data
    rs2::colorizer color_map;

    // Declare RealSense pipeline, encapsulating the actual device and sensors
    rs2::pipeline pipe;
    // Start streaming with default recommended configuration
    pipe.start();

    using namespace cv;
    const auto window_name = "Display Image";
    namedWindow(window_name, WINDOW_AUTOSIZE);

    while (getWindowProperty(window_name, WND_PROP_AUTOSIZE) >= 0)
    {
        rs2::frameset data = pipe.wait_for_frames(); // Wait for next set of frames from the camera
        rs2::frame depth = data.get_depth_frame().apply_filter(color_map);
        rs2::frame image_rs = data.get_color_frame();
        // Query frame size (width and height)
        const int w = image_rs.as<rs2::video_frame>().get_width();
        const int h = image_rs.as<rs2::video_frame>().get_height();

        // Create OpenCV matrix of size (w,h) from the colorized depth data
        std::vector<cv::Rect> features;
        Mat gray_img;
        Mat image(Size(w, h), CV_8UC3, (void*)image_rs.get_data(), Mat::AUTO_STEP);
        cv::cvtColor(image, image, cv::COLOR_RGB2BGR);
        cv::cvtColor(image, gray_img, cv::COLOR_BGR2GRAY);
        cv::equalizeHist(gray_img, gray_img);
        classifier.detectMultiScale(gray_img,features,1.1,2,0 | cv::CASCADE_SCALE_IMAGE, cv::Size(30, 30));
        for (auto&& feature : features) {
            cv::rectangle(image, feature, cv::Scalar(0, 255, 0), 2);
        }
        // Update the window with new data
        cv:;imshow(window_name, image);
        switch (cv::waitKey(1)) {
        case 'q':
        case 'Q':
            return EXIT_SUCCESS;
        case 's':
        case 'S':
            imwrite("result.jpg", image);
            return EXIT_SUCCESS;
        default:
            break;
        }
    }

    return EXIT_SUCCESS;
}
catch (const rs2::error & e)
{
    std::cerr << "RealSense error calling " << e.get_failed_function() << "(" << e.get_failed_args() << "):\n    " << e.what() << std::endl;
    return EXIT_FAILURE;
}
catch (const std::exception& e)
{
    std::cerr << e.what() << std::endl;
    return EXIT_FAILURE;
}