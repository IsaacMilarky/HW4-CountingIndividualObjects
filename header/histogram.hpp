#include <iostream>
#include <map>
#include <algorithm>
#include <cmath>
#include <opencv2/highgui.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/core/types_c.h>

//The two function required for the first part.
std::map<int,int> calculateRawHistogram(cv::Mat * ,int = 256);
std::map<int,int> calculateRawHistogram(cv::Mat *, int, cv::Mat *);


//Encapsulate the concept of an image's histogram data.
//Flesh out for each assignment.
//Takes a grayscale image as an argument which should be single channel mat of an image.
class ImageHistogram
{
    
public:
    std::map<int,int> rawHistogram;
    std::map<int,double> freqHistogram;
    unsigned int numPixels;
    unsigned int nbins;
    int max_intensity;
    int min_intensity;
    int max_domain;
    int min_domain;
    ImageHistogram(cv::Mat *, int = 256);
    ImageHistogram(cv::Mat *,int, cv::Mat *);
    //normalize, other stuff, etc..
    cv::Mat createHistogramImage();
    void linearStretch(int,int);
    void discardPercentage(unsigned int);
private:
    std::map<int,double> calcFrequency();
    std::map<int,int>::iterator getMaxHistogramIntensityScalar();
    std::map<int,int>::iterator getMinHistogramIntensityScalar();
    std::map<int,int>::iterator getMaxHistogramDomainScalar();
    std::map<int,int>::iterator getMinHistogramDomainScalar();
};

cv::Mat applyLinearStretch(cv::Mat *);
