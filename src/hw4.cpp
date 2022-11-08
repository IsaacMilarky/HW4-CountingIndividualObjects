#include <iostream>
#include <vector>
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

    //Create marker object for opencv watershed
    cv::Mat dist_8u;
    dist.convertTo(dist_8u, CV_8U);

    //Find total markers
    std::vector<std::vector< cv::Point> > contours;
    cv::findContours(dist_8u,contours,cv::RETR_EXTERNAL,cv::CHAIN_APPROX_SIMPLE);

    cv::Mat markers = cv::Mat::zeros(dist.size(),CV_32S);

    //Forground markers
    for (size_t iter = 0; iter < contours.size(); iter++)
    {
        cv::drawContours(markers,contours,static_cast<int>(iter), cv::Scalar(static_cast<int>(iter)+1), -1);
    }

    cv::circle(markers,cv::Point(5,5),3,cv::Scalar(255), -1);
    cv::Mat markers8u;

    markers.convertTo(markers8u, CV_8U,10);
    imwrite("marker_processed.png",markers8u);

    //Take Laplacian of gaussian of original image.
    cv::Mat LoGFilter = (cv::Mat_<float>(3,3) <<
              1,  1, 1,
              1, -8, 1,
              1,  1, 1);

    cv::Mat laplacianImage;
    cv::filter2D(inputImageColorGrouping, laplacianImage, CV_32F, LoGFilter);
    cv::Mat sharp;
    inputImageColorGrouping.convertTo(sharp,CV_32F);
    cv::Mat imgResult = sharp - laplacianImage; 

    //Convert back to rgb images.
    imgResult.convertTo(imgResult, CV_8UC3);
    laplacianImage.convertTo(laplacianImage, CV_8UC3);

    imwrite("Sharpened.png",imgResult);

    //Perform watershed.
    cv::watershed(imgResult,markers);

    cv::Mat mark;
    markers.convertTo(mark, CV_8U);
    cv::bitwise_not(mark, mark);
    //    imshow("Markers_v2", mark); // uncomment this if you want to see how the mark
    // image looks like at that point
    // Generate random colors
    std::vector<cv::Vec3b> colors;
    for (size_t i = 0; i < contours.size(); i++)
    {
        int b = cv::theRNG().uniform(0, 256);
        int g = cv::theRNG().uniform(0, 256);
        int r = cv::theRNG().uniform(0, 256);
        colors.push_back(cv::Vec3b((uchar)b, (uchar)g, (uchar)r));
    }
    // Create the result image
    cv::Mat dst = cv::Mat::zeros(markers.size(), CV_8UC3);
    // Fill labeled objects with random colors
    for (int i = 0; i < markers.rows; i++)
    {
        for (int j = 0; j < markers.cols; j++)
        {
            int index = markers.at<int>(i,j);
            if (index > 0 && index <= static_cast<int>(contours.size()))
            {
                dst.at<cv::Vec3b>(i,j) = colors[index-1];
            }
        }
    }
    // Visualize the final image
    imwrite("Final_Result.png", dst);
    return 0;
}