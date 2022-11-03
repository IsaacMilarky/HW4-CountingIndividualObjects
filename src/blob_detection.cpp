#include "../header/blob_detection.hpp"


cv::Mat createLoGKernel(double standard_deviation, int kernelSize)
{
    //Create raw array of floats for kernel
    //std::unique_ptr<float[]> rawKernelData = std::make_unique<float[]>(kernelSize);

    //I know its bad to not use smart pointers but opencv wouldn't take a unique_ptr
    float * rawKernelData = new float[kernelSize * kernelSize];

    //Calculate radius of kernel.
    int radius = std::ceil(static_cast<float>(kernelSize) / 2.0);
    int xOrigin = 0;
    int yOrigin = 0;

    int yMin = (yOrigin - (radius - 1));
    int xMin = (xOrigin - (radius - 1));
    int xMax = (xOrigin + radius);
    int yMax = (yOrigin + radius);

    int cursor = 0;

    const double EULER_CONSTANT = std::exp(1.0);

    std::cout << (xMax - xMin) << "x" << (yMax - yMin) << std::endl;

    //Iterate through kernelSize x kernelSize grid.
    for(int xIter = xMin; xIter < xMax; xIter++)
    {
        for(int yIter = yMin; yIter < yMax; yIter++)
        {
            //rawKernelData[cursor]
            //Define first term of approximation equation.
            double firstTerm = -1.0 / (PI * std::pow(standard_deviation,4));

            double secondTerm = 1 - ( (std::pow(xIter,2) + std::pow(yIter,2)) / (2 * std::pow(standard_deviation,2)) );

            double thirdTerm = std::pow(EULER_CONSTANT, -1 * ((std::pow(xIter,2) + std::pow(yIter,2)) / (2 * std::pow(standard_deviation,2))));

            rawKernelData[cursor] = firstTerm * secondTerm * thirdTerm * CONSTANT_CORRECTION_FACTOR * standard_deviation;
            std::cout << rawKernelData[cursor] << ", ";
            cursor++;
        }

        std::cout << std::endl;
    }

    cv::Mat toReturn = cv::Mat(kernelSize,kernelSize, CV_32F,rawKernelData);
    delete rawKernelData;
    return toReturn.clone();
}

cv::Mat applyKmeans(cv::Mat * src, int k)
{
    const int MAX_ITER = 4;
    const unsigned int arrSize = src->rows * src->cols;

    //Data needs to be 1d for opencv
    cv::Mat data = src->reshape(1,arrSize);
    data.convertTo(data, CV_32F);

    //Store color of each cluster and status of each pixel after kmeans
    std::vector<int> labels;
    cv::Mat1f colors;

    //Apply kmeans
    cv::kmeans(data,k, labels,
        cv::TermCriteria(cv::TermCriteria::EPS+cv::TermCriteria::COUNT, 10, 1.),
            MAX_ITER, cv::KMEANS_PP_CENTERS, colors);

    //Transform pixels to cluster color
    for (unsigned int iter = 0 ; iter < arrSize ; iter++ )
    {
		data.at<float>(iter, 0) = colors(labels[iter], 0);
		data.at<float>(iter, 1) = colors(labels[iter], 1);
		data.at<float>(iter, 2) = colors(labels[iter], 2);
	}

    cv::Mat outputImage = data.reshape(3,src->rows);
    outputImage.convertTo(outputImage,CV_8U);

    return outputImage.clone();
}

cv::Mat applyKmeansGreyScale(cv::Mat * src, int k)
{
    const int MAX_ITER = 4;
    const unsigned int arrSize = src->rows * src->cols;

    //Data needs to be 1d for opencv
    cv::Mat data = src->reshape(1,arrSize);
    data.convertTo(data, CV_32FC1);

    //Store color of each cluster and status of each pixel after kmeans
    std::vector<int> labels;
    cv::Mat1f colors;

    //Apply kmeans
    cv::kmeans(data,k, labels,
        cv::TermCriteria(cv::TermCriteria::EPS+cv::TermCriteria::COUNT, 10, 1.),
            MAX_ITER, cv::KMEANS_PP_CENTERS, colors);

    //Transform pixels to cluster color
    for (unsigned int iter = 0 ; iter < arrSize ; iter++ )
    {
		data.at<float>(iter, 0) = colors(labels[iter], 0);
		data.at<float>(iter, 1) = colors(labels[iter], 1);
		data.at<float>(iter, 2) = colors(labels[iter], 2);
	}

    cv::Mat outputImage = data.reshape(1,src->rows);
    outputImage.convertTo(outputImage,CV_8UC1);

    return outputImage.clone();
}

cv::Mat getKmeansBinMask(cv::Mat * src)
{
    const int MAX_ITER = 4;
    const unsigned int arrSize = src->rows * src->cols;

    cv::Size inputSize = src->size();
    cv::Mat outputGreyscale = cv::Mat(inputSize,CV_8U);

    //Data needs to be 1d for opencv
    cv::Mat greyData = outputGreyscale.reshape(1,arrSize);
    cv::Mat data = src->reshape(1,arrSize);
    data.convertTo(data, CV_32F);

    //Store color of each cluster and status of each pixel after kmeans
    std::vector<int> labels;
    cv::Mat1f colors;

    //Apply kmeans
    cv::kmeans(data,2, labels,
        cv::TermCriteria(cv::TermCriteria::EPS+cv::TermCriteria::COUNT, 10, 1.),
            MAX_ITER, cv::KMEANS_PP_CENTERS, colors);

    //Transform color to mask colors
    float firstColorGrey = 0.2989 * colors(0,2) + 0.5871 * colors(0,1) + 0.114 * colors(0,0);
    float secondColorGrey = 0.2989 * colors(1,2) + 0.5871 * colors(1,1) + 0.114 * colors(1,0);

    if(firstColorGrey < secondColorGrey)
    {
        firstColorGrey = 0;
        secondColorGrey = 255;
    }
    else
    {
        secondColorGrey = 0;
        firstColorGrey = 255;
    }

    float greyColors[2] = {firstColorGrey,secondColorGrey};

    //Transform pixels to cluster color
    for (unsigned int iter = 0 ; iter < arrSize ; iter++ )
    {
		//data.at<float>(iter, 0) = colors(labels[iter], 0);
		//data.at<float>(iter, 1) = colors(labels[iter], 1);
		//data.at<float>(iter, 2) = colors(labels[iter], 2);
        greyData.at<uchar>(iter) = greyColors[labels[iter]];
	}

    //std::cout << colors << std::endl;
    cv::Mat outputImage = greyData.reshape(1,src->rows);
    outputImage.convertTo(outputImage,CV_8UC1);
    return outputImage.clone();
}