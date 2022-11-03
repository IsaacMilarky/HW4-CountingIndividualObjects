#include "../header/mean_squared.hpp"

double mean_squared_error(cv::Mat * firstImage, cv::Mat * secondImage)
{
    //Do the summation on the intersection of the images.

    double sum = 0;
    unsigned int width = firstImage->size().width > secondImage->size().width ? firstImage->size().width : secondImage->size().width;
    unsigned int height = firstImage->size().height > secondImage->size().height ? firstImage->size().height : secondImage->size().height;
    //Use MSE formula.
    for(unsigned int xCoord = 0; xCoord < width; xCoord++)
    {
        for(unsigned int yCoord = 0; yCoord < height; yCoord++)
        {
            double first = 0;
            //Grab values and bounds check.
            if(xCoord < firstImage->size().width && yCoord < firstImage->size().height)
            {
                first = firstImage->at<uchar>(yCoord,xCoord);
            }

            double second = 0;
            if(xCoord < secondImage->size().width && yCoord < secondImage->size().height)
            {
                second = secondImage->at<uchar>(yCoord,xCoord);
            }

            sum += std::pow(first - second, 2);
            //std::cout << sum << std::endl;
        }
    }

    return (sum / (static_cast<double>(width * height)));
}