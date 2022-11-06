#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include "../header/blob_detection.hpp"
#include "../header/histogram.hpp"

int main(int argc, char ** argv)
{
    if (argc != 2)
    {
        std::cerr << "Incorrect number of args! \nUsage: ./kmeans <input>" << std::endl;
        return -1;
    }

    int k = 2;
    cv::Mat inputImageColorGrouping = cv::imread(argv[1], cv::IMREAD_COLOR);

    cv::Mat mask = getKmeansBinMask(&inputImageColorGrouping);
    imwrite("kmeansColor.png", mask);

    //Get distance transform for markers.
    cv::Mat dist;
    cv::distanceTransform(mask,dist,cv::DIST_L2,3);
    imwrite("distancetransform.png",dist);
    //cv::normalize(dist,dist,0,1.0,cv::NORM_MINMAX);
    //Threshold distance transform at 10 for markers.
    cv::threshold(dist,dist,8,255.0,cv::THRESH_BINARY);

    //Dilate markers to be more visible.
    cv::Mat kernel1 = cv::Mat::ones(3, 3, CV_8U);
    dilate(dist, dist, kernel1);
    imwrite("markers.png", dist);

    return 0;
}