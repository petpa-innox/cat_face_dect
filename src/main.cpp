#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>

using namespace std;
using namespace cv;

int main(int argc, char** argv)
{
    
    if (argc != 3)
    {
        cout << "Usage: opencv_test <image path> <classifier path>" << endl;
        return -1;
    }
    const char* const clasifier_path = "/home/holmes/code/cat_face_detection/haarcascade_frontalcatface.xml";
    cv::CascadeClassifier  classifier(argv[2]);
    char *imgName = argv[1];
    Mat image;

    image = imread(imgName, 1);
    if (!image.data)
    {
        cout << "No image data" << endl;
        return -1;
    }
    
    Mat gray_img;
    std::vector<cv::Rect> features;
    
    cvtColor(image, gray_img, cv::COLOR_BGR2GRAY);
    equalizeHist(gray_img, gray_img);
    classifier.detectMultiScale(gray_img,features,1.1,2,0 | cv::CASCADE_SCALE_IMAGE, cv::Size(30, 30));
    
    for (auto&& feature : features) {
      cv::rectangle(image, feature, cv::Scalar(0, 255, 0), 2);
    }

    const char* const window_name{"Facial Recognition Window"};
    cv::namedWindow(window_name, cv::WINDOW_AUTOSIZE);
    cv::imshow(window_name, image);

    switch (cv::waitKey()) {
    case 'q':
    case 'Q':
        std::exit(EXIT_SUCCESS);
    case 's':
    case 'S':
        imwrite("result.jpg", image);
        std::exit(EXIT_SUCCESS);
    default:
        break;
    }
    return 0;
}