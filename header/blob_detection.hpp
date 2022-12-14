#include <iostream>
#include <algorithm>
#include <memory>
#include <cmath>
#include <vector>
#include <fstream>
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/core/utility.hpp"

#define PI 3.14159
#define CONSTANT_CORRECTION_FACTOR 48000.0

cv::Mat createLoGKernel(double standard_deviation,int kernelSize);

cv::Mat applyKmeans(cv::Mat * src, int k);

cv::Mat applyKmeansGreyScale(cv::Mat * src, int k);

cv::Mat getKmeansBinMask(cv::Mat * src);

cv::Mat getPreMarkers(cv::Mat * src);

cv::Mat watershedHighlightObjects(cv::Mat * src, cv::Mat * markerMask);

cv::Mat applyComponentLabeling(cv::Mat * src, cv::Mat * origin);