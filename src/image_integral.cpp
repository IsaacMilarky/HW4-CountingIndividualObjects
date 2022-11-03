#include "../header/image_integral.hpp"

IntegralImage::IntegralImage(cv::Mat * srcImage)
{
    
    height = srcImage->size().height;
    width = srcImage->size().width;
    //Initialize 2d vector of unsigned ints
    integralMatrix = std::vector<std::vector<unsigned int> >(
        width + 1,
        std::vector<unsigned int>(height + 1));

    //Compute Integral Image
    //I(x,y) = i(x,y) + I(x,y-1) + I(x-1,y-1) + I(x-1,y)
    for(int xCoord = 0; xCoord < width; xCoord++)
    {
        for(int yCoord = 0; yCoord < height; yCoord++)
        {
            int aboveIntensity, upLeftIntensity;
            //Deal with oob
            if((yCoord-1) < 0)
            {
                aboveIntensity = 0;
                //upLeftIntensity = 0;
            }
            else
            {
                aboveIntensity = integralMatrix[xCoord][yCoord - 1];
            }

            int leftIntensity;
            if((xCoord - 1) < 0)
            {
                leftIntensity = 0;
            }
            else
            {
                leftIntensity = integralMatrix[xCoord - 1][yCoord];
            }


            if((xCoord -1) < 0 || (yCoord -1) < 0)
            {
                upLeftIntensity = 0;
            }
            else
            {
                upLeftIntensity = integralMatrix[xCoord -1][yCoord -1];
            }
            //i(x,y)
            int rawImagePixel = srcImage->at<uchar>(yCoord,xCoord);
            //std::cout << (int)*(uchar*)(void *)&srcImage->at<uchar>(yCoord,xCoord) << std::endl;
            //Be mindful of zeroes out of bounds.
            integralMatrix[xCoord][yCoord] = rawImagePixel + aboveIntensity + upLeftIntensity + leftIntensity;
        }
    }
}

unsigned int IntegralImage::at(int x, int y)
{
    unsigned int xCoord = std::clamp(x,0,width);
    unsigned int yCoord = std::clamp(y,0,height);

    return integralMatrix[xCoord][yCoord];
}