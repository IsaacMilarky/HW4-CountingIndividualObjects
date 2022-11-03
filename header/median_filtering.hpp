#include "image_integral.hpp"
#include <set>
#include <queue>
#include <cmath>
#include <memory>


class MedianImageFilter
{
    public:
        //Create a filter with max size of int
        MedianImageFilter(int, cv::Mat *,unsigned int, unsigned int);
        //~MedianImageFilter();
        //Take the pointer to a pixel and add it to the set.
        void addPixelToSet(uchar);
        int getZMed();
        //Increase window size by one
        void increaseWindowSize();
        int minVal, maxVal;
        int size;
        int centerVal;
    private:
        //Left and right heaps for insertion to find median.
        std::priority_queue <uchar, std::vector<uchar> > left;
        std::priority_queue <uchar, std::vector<uchar>, std::greater<uchar> > right;
        unsigned int xPos;
        unsigned int yPos;
        cv::Mat * imagePtr;
        //void * zeroMemory;
        int xMin,yMin,xMax,yMax;
};

cv::Mat adaptive_median_filter(cv::Mat *,int);

int adaptive_median_filter_stageA(MedianImageFilter&,int);
int adaptive_median_filter_stageB(MedianImageFilter&,int);
