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

    //Declare input image and erosion/dilation kernel
    cv::Mat inputImageColorGrouping = cv::imread(argv[1], cv::IMREAD_COLOR);
    cv::Mat kernel1 = cv::Mat::ones(3, 3, CV_8U);

    //Get raw mask that will be turned into marker objects (markers and contours)
    cv::Mat preMarkers = getPreMarkers(&inputImageColorGrouping);

    //Use mask found to create marker objects and apply watershed. Saving it to dst.
    cv::Mat dst = watershedHighlightObjects(&inputImageColorGrouping,&preMarkers);

    // Visualize the watershed image
    imwrite("watershed.png", dst);

    //Get background from kmeans
    cv::Mat mask = getKmeansBinMask(&inputImageColorGrouping);

    //Write out component labels and original image with the blue parts highlighted.
    imwrite("componentLabeling.png",applyComponentLabeling(&preMarkers,&inputImageColorGrouping));
    imwrite("bluHighlighted.png", inputImageColorGrouping);
    return 0;
}