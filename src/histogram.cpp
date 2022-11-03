#include "../header/histogram.hpp"


std::map<int,int> calculateRawHistogram(cv::Mat * grayImage,int nbins)
{
    std::map<int,int> rawHist;


    //Compute histogram and return 
    for(unsigned int xCoord = 0; xCoord < grayImage->size().width; xCoord++)
    {
        for(unsigned int yCoord = 0; yCoord < grayImage->size().height; yCoord++)
        {
            //Increment value
            rawHist[grayImage->at<uchar>(yCoord,xCoord)]++;
        }
    }


    return rawHist;
}

//Overload of function with a provided mask for the assignment.
std::map <int,int> calculateRawHistogram(cv::Mat * grayImage, int nbins, cv::Mat * maskImage)
{
    std::map<int,int> rawHist;


    //Compute histogram and return 
    for(unsigned int xCoord = 0; xCoord < grayImage->size().width; xCoord++)
    {
        for(unsigned int yCoord = 0; yCoord < grayImage->size().height; yCoord++)
        {
            //std::cout << "Mask Status: " << (int)maskImage->at<uchar>(yCoord,xCoord) << std::endl;
            //Increment value
            if((int)maskImage->at<uchar>(yCoord,xCoord) == 255)
                rawHist[grayImage->at<uchar>(yCoord,xCoord)]++;
        }
    }


    return rawHist;
}


ImageHistogram::ImageHistogram(cv::Mat * targetImage, int nbins)
{
    rawHistogram = calculateRawHistogram(targetImage,nbins);
    numPixels = targetImage->total();
    freqHistogram = calcFrequency();
    this->nbins = nbins;
    //std::cout << "Creating ImageHistogram" << numPixels << std::endl;

    //double sum = 0.0;
    //Calculate frequency histogram
    max_intensity = getMaxHistogramIntensityScalar()->second;
    min_intensity = getMinHistogramIntensityScalar()->second;
    max_domain = getMaxHistogramDomainScalar()->first;
    min_domain = getMinHistogramDomainScalar()->first;
}

ImageHistogram::ImageHistogram(cv::Mat * targetImage, int nbins, cv::Mat * maskImage)
{
    numPixels = 0;
    rawHistogram = calculateRawHistogram(targetImage,nbins,maskImage);

    //Count the pixels left behind by the mask
    for(auto &iter: rawHistogram)
    {
        numPixels += iter.second;
    }

    freqHistogram = calcFrequency();
    this->nbins = nbins;
    max_intensity = getMaxHistogramIntensityScalar()->second;
    min_intensity = getMinHistogramIntensityScalar()->second;
    max_domain = getMaxHistogramDomainScalar()->first;
    min_domain = getMinHistogramDomainScalar()->first;
}

cv::Mat ImageHistogram::createHistogramImage()
{
    int minVal = min_intensity;
    int maxVal = max_intensity;//getMaxHistogramValue(rawHistogram);

    //Make a bunch of constants for the histogram image including
    //Size of histogram image to save
    const cv::Scalar bgColor(0,0,0); //black
    const cv::Scalar lineColor(0, 0, 255); //red
    const int height = 1024;
    const int width = 1024;
    const int min_y = 0;
    const int max_y = height - min_y;
    float bin_width   = static_cast<float>(width) / static_cast<float>(nbins);
    cv::Mat toReturn(height, width, CV_8UC3, bgColor); 
    const short enhance_line_contrast = 10;

    for(unsigned int iter = 1; iter < nbins; iter++)
    {
        //Define Slope lines between points of the histogram with two points
        int x1 = std::round(bin_width * (iter - 1));
        int x2 = std::round(bin_width * (iter));

        //Clamp vertical values since they aren't pre-determined by the image width.
        //Multiply by constant because otherwise they aren't as easy to see when plotted to an image.
        int y1 = std::clamp(height - static_cast<int>(std::round(freqHistogram[(iter - 1)] * height * enhance_line_contrast)) , min_y, max_y);
        int y2 = std::clamp(height - static_cast<int>(std::round(freqHistogram[iter] * height * enhance_line_contrast)) , min_y, max_y);

        //std::cout << "(" << x1 << "," << freqHistogram[(iter - 1)] * height * enhance_line_contrast << ")" << std::endl;
        cv::line(toReturn, cv::Point(x1, y1), cv::Point(x2, y2), lineColor, 1, cv::LINE_AA);
    }

    return toReturn.clone();
}

//Aply linear stretch algorithm to the object
void ImageHistogram::linearStretch(int nmin, int nmax)
{
    //calculate coefficient to stretch domain of new histogram.
    int domainDiff = max_domain - min_domain;
    float space = static_cast<float>(nmax - nmin) / static_cast<float>(domainDiff);
    int original_nbins = domainDiff;

    //NewHistogram(j) = nmin + (j * space) for all bins j
    std::map<int,int> newRawHist;

    for(unsigned int j = 0; j <= original_nbins; j++)
    {
        newRawHist[nmin + static_cast<int>(j * space)] = rawHistogram[j];
        //std::cout << "Old value: (" << j << "," << rawHistogram[j] << ")\n";
        //std::cout << "New value: (" << nmin + static_cast<int>(j * space) << "," << newRawHist[nmin + static_cast<int>(j * space)] << ")\n";
    }

    //Overwrite old histogram values.
    rawHistogram = newRawHist;
    freqHistogram = calcFrequency();
    nbins = nmax - nmin;
    max_intensity = getMaxHistogramIntensityScalar()->second;
    min_intensity = getMinHistogramIntensityScalar()->second;
    max_domain = nmax;
    min_domain = nmin;
}

void ImageHistogram::discardPercentage(unsigned int percentage)
{
    //First, compute how many pixels the percentage actually is.
    int pixelsToRemove = std::round(static_cast<double>(numPixels) * (static_cast<float>(percentage) / 100));

    //Remove from front and back of the histogram equally
    int histogram_front = min_domain;
    int histogram_back = max_domain;

    //Then start removing pixels from the histogram
    while(pixelsToRemove > 0)
    {
        //Make sure that there are pixels that we can take away in this bin
        while(rawHistogram[histogram_front] <= 0)
        {
            histogram_front++;
            
            //Handle if no more pixels to take away
            if(histogram_front == (max_domain + 1))
            {
                return;
            }
        }

        //Same but for the rear bin
        while(rawHistogram[histogram_back] <= 0)
        {
            histogram_back--;

            //This should raise an exception but,
            //Its late and its not required for points so i'll do it later.
            if(histogram_back == (min_domain - 1))
            {
                return;
            }
        }

        //actually remove data from the histogram here.
        rawHistogram[histogram_back]--;
        pixelsToRemove--;

        //Deal with odd numbers
        if(pixelsToRemove > 0)
        {
            rawHistogram[histogram_front]--;
            pixelsToRemove--;
        }

    }

    freqHistogram = calcFrequency();
    nbins = histogram_back - histogram_front;
    max_intensity = getMaxHistogramIntensityScalar()->second;
    min_intensity = getMinHistogramIntensityScalar()->second;
    max_domain = histogram_back;
    min_domain = histogram_front;
}


std::map<int,double> ImageHistogram::calcFrequency()
{
    std::map<int,double> freqMap;
    for(auto &iter: rawHistogram)
    {
        freqMap[iter.first] = static_cast<double>(iter.second) / static_cast<double>(numPixels);

        //std::cout << iter.first << " : " << freqMap[iter.first] << std::endl;
        //sum += freqHistogram[iter.first];
    }

    return freqMap;
}

std::map<int,int>::iterator ImageHistogram::getMaxHistogramIntensityScalar()
{
    auto x = std::max_element(rawHistogram.begin(),rawHistogram.end(),
        [](const std::pair<int, int>& p1, const std::pair<int, int>& p2) {
            return p1.second < p2.second; });

    return x;
}

std::map<int,int>::iterator ImageHistogram::getMinHistogramIntensityScalar()
{
    auto x = std::min_element(rawHistogram.begin(),rawHistogram.end(),
        [](const std::pair<int, int>& p1, const std::pair<int, int>& p2) {
            return p1.second < p2.second; });

    return x;
}

std::map<int,int>::iterator ImageHistogram::getMaxHistogramDomainScalar()
{
    auto x = std::max_element(rawHistogram.begin(),rawHistogram.end(),
        [](const std::pair<int, int>& p1, const std::pair<int, int>& p2) {
            return p1.first < p2.first; });

    return x;
}

std::map<int,int>::iterator ImageHistogram::getMinHistogramDomainScalar()
{
    auto x = std::min_element(rawHistogram.begin(),rawHistogram.end(),
        [](const std::pair<int, int>& p1, const std::pair<int, int>& p2) {
            return p1.first < p2.first; });

    return x;
}

cv::Mat applyLinearStretch(cv::Mat * inputGreyScaleImage)
{
    //Create histogram of input
    ImageHistogram inputHist(inputGreyScaleImage);

    //Copy histogram object and stretch it
    ImageHistogram after = inputHist;
    after.linearStretch(0,256);

    //Now use differances between resulting histograms to transform image
    cv::Mat stretched = inputGreyScaleImage->clone();

    //Find coefficiants
    int domainDiff = inputHist.max_domain - inputHist.min_domain;
    float space = 256 / static_cast<float>(domainDiff);
    int original_nbins = domainDiff;


    //Transform each pixel
    for(unsigned int xCoord = 0; xCoord < stretched.size().width; xCoord++)
    {
        for(unsigned int yCoord = 0; yCoord < stretched.size().height; yCoord++)
        {
            stretched.at<uchar>(yCoord,xCoord) = static_cast<int>(space * inputGreyScaleImage->at<uchar>(yCoord,xCoord));
        }
    }

    //Return complete mat image.
    return stretched.clone();
}