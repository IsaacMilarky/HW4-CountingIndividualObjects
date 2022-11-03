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

    imwrite("kmeansColor.png",getKmeansBinMask(&inputImageColorGrouping));

    cv::Mat inputImageIntensityGrouping = cv::imread(argv[1], cv::IMREAD_GRAYSCALE);

    imwrite("kmeansGrayscale.png",applyKmeansGreyScale(&inputImageIntensityGrouping,k));

    return 0;
}