#include <iostream>
#include <sstream>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include "../header/blob_detection.hpp"
#include "../header/histogram.hpp"

int main(int argc, char ** argv)
{
    if (argc != 3)
    {
        std::cerr << "Incorrect number of args! \nUsage: ./kmeans <input> <k-arg>" << std::endl;
        return -1;
    }

    //Parse arguments
    int k;
    std::stringstream kArgConvert;
    kArgConvert << argv[2];
    kArgConvert >> k;

    cv::Mat inputImageColorGrouping = cv::imread(argv[1], cv::IMREAD_COLOR);

    imwrite("kmeansColor.png",applyKmeans(&inputImageColorGrouping,k));

    cv::Mat inputImageIntensityGrouping = cv::imread(argv[1], cv::IMREAD_GRAYSCALE);

    imwrite("kmeansGrayscale.png",applyKmeansGreyScale(&inputImageIntensityGrouping,k));

    return 0;
}