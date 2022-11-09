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
    float firstColorGrey = 0;//0.2989 * colors(0,2) + 0.5871 * colors(0,1) + 0.114 * colors(0,0);
    float secondColorGrey = 255;//0.2989 * colors(1,2) + 0.5871 * colors(1,1) + 0.114 * colors(1,0);

    //if(firstColorGrey < secondColorGrey)
    //{
    //    firstColorGrey = 0;
    //    secondColorGrey = 255;
    //}
    //else
    //{
    //    secondColorGrey = 0;
    //    firstColorGrey = 255;
    //}

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

cv::Mat getPreMarkers(cv::Mat * src)
{
    cv::Mat kernel1 = cv::Mat::ones(3, 3, CV_8U);

    //Get background from kmeans
    cv::Mat mask = getKmeansBinMask(src);
    imwrite("kmeansMask.png", mask);

    //Get distance transform for sure foregound..
    cv::Mat dist;
    cv::distanceTransform(mask,dist,cv::DIST_MASK_3,0);
    imwrite("distancetransform.png",dist);

    //Threshold distance transform at 1 for markers.
    cv::threshold(dist,dist,1,255.0,cv::THRESH_BINARY);
    
    //erode foreground to be more distinct.
    cv::Mat sureForeground;
    erode(dist, sureForeground, kernel1, cv::Point(-1, -1), 4);
    imwrite("foreground.png", sureForeground);

    //Dilate mask image for more generous background detection.
    dilate(mask,mask,kernel1);
    imwrite("dilation.png",mask);

    //Invert the kmeans mask to find background highlighted.
    sureForeground.convertTo(sureForeground,CV_8UC1);
    cv::bitwise_not(mask,mask);

    //Compute foreground and background markers into one image.
    cv::Mat preMarkers = mask + sureForeground;
    imwrite("unsure.png",preMarkers);   
    preMarkers.convertTo(preMarkers,CV_8UC3);

    return preMarkers.clone();
}

cv::Mat watershedHighlightObjects(cv::Mat * src, cv::Mat * markerMask)
{
    //Create marker object for opencv watershed
    cv::Mat dist_8u;
    markerMask->convertTo(dist_8u, CV_8U);

    //Find total markers
    std::vector<std::vector< cv::Point> > contours;
    cv::findContours(dist_8u,contours,cv::RETR_TREE,cv::CHAIN_APPROX_NONE);

    cv::Mat markers = cv::Mat::zeros(markerMask->size(),CV_32S);

    //Forground markers
    for (size_t iter = 0; iter < contours.size(); iter++)
    {
        cv::drawContours(markers,contours,static_cast<int>(iter), cv::Scalar(static_cast<int>(iter)+1), -1);
    }

    //cv::circle(markers,cv::Point(5,5),3,cv::Scalar(255), -1);
    cv::Mat markers8u;

    markers.convertTo(markers8u, CV_8U,10);
    imwrite("marker_processed.png",markers8u);

    //Take Laplacian of gaussian of original image.
    cv::Mat LoGFilter = (cv::Mat_<float>(3,3) <<
              1,  1, 1,
              1, -8, 1,
              1,  1, 1);

    cv::Mat laplacianImage;
    cv::filter2D(*src, laplacianImage, CV_32F, LoGFilter);
    cv::Mat sharp;
    src->convertTo(sharp,CV_32F);
    cv::Mat imgResult = sharp - laplacianImage; 

    //Convert back to rgb images.
    imgResult.convertTo(imgResult, CV_8UC3);
    laplacianImage.convertTo(laplacianImage, CV_8UC3);

    imwrite("Sharpened.png",imgResult);

    //Perform watershed.
    cv::watershed(imgResult,markers);

    cv::Mat mark;
    markers.convertTo(mark, CV_8U);
    cv::bitwise_not(mark, mark);
    //imwrite("Markers_v2.png", mark); // uncomment this if you want to see how the mark
    // image looks like at that point
    // Generate random colors
    std::vector<cv::Vec3b> colors;
    for (size_t i = 0; i < contours.size(); i++)
    {
        int b = cv::theRNG().uniform(0, 256);
        int g = cv::theRNG().uniform(0, 256);
        int r = cv::theRNG().uniform(0, 256);
        colors.push_back(cv::Vec3b((uchar)b, (uchar)g, (uchar)r));
    }
    // Create the result image
    cv::Mat dst = cv::Mat::zeros(markers.size(), CV_8UC3);
    // Fill labeled objects with random colors
    for (int i = 0; i < markers.rows; i++)
    {
        for (int j = 0; j < markers.cols; j++)
        {
            int index = markers.at<int>(i,j);
            if (index > 0 && index <= static_cast<int>(contours.size()))
            {
                dst.at<cv::Vec3b>(i,j) = colors[index-1];
            }
        }
    }

    return dst.clone();
}

cv::Mat applyComponentLabeling(cv::Mat * watershed, cv::Mat * original)
{
    cv::Mat labelImage(watershed->size(), CV_32S);

    cv::Mat stats;
    cv::Mat centroids;
    int nLabels = cv::connectedComponentsWithStats(*watershed,labelImage,stats,centroids);

    //Normalize and visualize connected components.
    cv::Mat seeLabels;
    cv::normalize(labelImage,seeLabels,0,255,cv::NORM_MINMAX,CV_8U);

    //Compute area and mean color for each.
    std::cout << nLabels << std::endl;

    std::cout << "Number of regions: " << stats.size() << std::endl;
    //std::cout << centroids << std::endl;
    
    for(int i=0; i<stats.rows; i++)
    {
        //Basic stats for testing.
        int area = stats.at<int>(cv::Point(cv::CC_STAT_AREA, i));
        //int y = stats.at<int>(cv::Point(1, i));
        //int w = stats.at<int>(cv::Point(2, i));
        //int h = stats.at<int>(cv::Point(3, i));

        std::cout << "Area in pixels for region " << i << " = " << area << std::endl;
      
    }

    //Find average color of each region.
    //The easy way with a lot of memory.
    std::vector<double> bSumColors(stats.rows, 0.0);
    std::vector<double> gSumColors(stats.rows, 0.0);
    std::vector<double> rSumColors(stats.rows, 0.0);

    std::vector<double> pixelSums(stats.rows, 0.0);


    for(int rows = 0; rows < original->rows; ++rows)
    {
        for(int cols = 0; cols < original->cols; ++cols){
            int label = labelImage.at<int>(rows, cols);
            cv::Vec3b pixel = original->at<cv::Vec3b>(rows, cols);
            //std::cout << label << std::endl;
            
            //Update value sums
            bSumColors.at(label) += pixel[0];
            gSumColors.at(label) += pixel[1];
            rSumColors.at(label) += pixel[2];

            pixelSums.at(label)++;

            //std::cout << pixelSums.at(label) << std::endl;
         }
    }

    //Calculate averages.
    for(int labelIter = 0; labelIter < stats.rows; labelIter++)
    {
        bSumColors.at(labelIter) /= pixelSums.at(labelIter);
        gSumColors.at(labelIter) /= pixelSums.at(labelIter);
        rSumColors.at(labelIter) /= pixelSums.at(labelIter);

        std::cout << "Mean color for region " << labelIter << " : (" << rSumColors.at(labelIter) << ", " << bSumColors.at(labelIter) << ", " << gSumColors.at(labelIter) << ") \n"; 
    }

    return seeLabels.clone();
}