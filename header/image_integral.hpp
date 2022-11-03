#include "histogram.hpp"
#include <vector>

//Define integral image and integral histogram classes and functions here.
class IntegralImage
{
    public:

        IntegralImage(cv::Mat *);

        unsigned int at(int, int);
    
    private:
        std::vector<std::vector<unsigned int> > integralMatrix;
        int height;
        int width;
        
};