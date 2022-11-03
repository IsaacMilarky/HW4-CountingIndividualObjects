#include "../header/median_filtering.hpp"


MedianImageFilter::MedianImageFilter(int windowSize, cv::Mat * target, unsigned int x, unsigned int y)
{
    size = windowSize;
    xPos = x;
    yPos = y;
    imagePtr = target;

    //Get radius of window.
    //3 goes to 2
    //5 goes to 3
    //7 goes to 4
    //9 goes to 5 etc..
    int radius = std::ceil(static_cast<float>(size) / 2.0);
    xMin = (xPos - (radius - 1));
    yMin = (yPos - (radius - 1));
    xMax = (xPos + radius);
    yMax = (yPos + radius);

    auto targetSize = target->size();
    //Iterate over window and add relevant pixels
    for(int xIter = xMin; xIter < xMax; xIter++)
    {
        for(int yIter = yMin; yIter < yMax; yIter++)
        {
            //Detect OOB
            if(xIter < 0 || yIter < 0 || xIter >= targetSize.width || yIter >= targetSize.height)
            {
                //add dummy 0 value
                addPixelToSet(0);
            }
            else
            {
                //Add if not oob.
                addPixelToSet(target->at<uchar>(yIter,xIter));
            }
        }
    }

    centerVal = target->at<uchar>(y,x);
}


void MedianImageFilter::addPixelToSet(uchar x)
{
    //If this is the first element to insert insert left.
    if(left.size() == 0)
    {
        left.push(x);
        minVal = x;
        maxVal = x;
        return;
    }

    uchar rootLeft = left.top();
    

    //If the new element x is smaller than the root of Left then we insert x to Left
    if(x < rootLeft)
    {
        left.push(x);
    }
    else
    {
        right.push(x);
    }

    /*
        Balance the heaps (after this step heaps will be either balanced or
        one of them will contain 1 more item)

        if number of elements in one of the heaps is greater than the other by
        more than 1, remove the root element from the one containing more elements and
        add to the other one
   */
   unsigned int heapDiff = std::abs(static_cast<int>(left.size() - right.size()));

   while(heapDiff > 1)
   {
        uchar toPop;
        //Take root off of left.
        if(left.size() < right.size())
        {
            toPop = right.top();
            right.pop();
            left.push(toPop);
        }
        //Take root off of right
        else
        {
            toPop = left.top();
            left.pop();
            right.push(toPop);
        }


        heapDiff = std::abs(static_cast<int>(left.size() - right.size()));
   }

    if( x < minVal)
    {
        minVal = x;
    }
    else if ( x > maxVal)
    {
        maxVal = x;
    }
}

int MedianImageFilter::getZMed()
{
    int rootLeft = left.top();
    
    /*
    std::cout << std::endl;
    while(left.empty() == false)
    {
        std::cout << (int)*(uchar*)left.top() << " ";
        left.pop();
    }
    std::cout << std::endl;
    while(right.empty() == false)
    {
        std::cout << (int)*(uchar*)right.top() << " ";
        right.pop();
    }
    std::cout << std::endl;
    */
    return rootLeft;
}

void MedianImageFilter::increaseWindowSize()
{
    //Increase window size by two
    //Increase by one isn't allowed bc then the window 
    //would be off center.
    int oldRadius = std::ceil(static_cast<float>(size) / 2.0);
    size += 2;
    int radius = std::ceil(static_cast<float>(size) / 2.0);
    auto targetSize = imagePtr->size();
    //Iterate over new pixels
    //(xMin--,y)
    //(x,yMin--)
    //(xMax++, y)
    //(x,yMax++)
    //(xMin--,yMin--)
    //(xMax++,yMax++)
    //(xMin--, yMax++)
    //(xMax++,yMin--)

    for(int yIter = yMin; yIter < yMax; yIter++)
    {
        //Detect OOB
        if((xMin - 1) < 0 || yIter < 0 || (xMin - 1) >= targetSize.width || yIter >= targetSize.height)
        {
            //add dummy 0 value
            addPixelToSet(0);
        }
        else
        {
            //Add if not oob.
            addPixelToSet(imagePtr->at<uchar>(yIter,(xMin - 1)));
        }

        //Detect OOB
        if((xMax + 1) < 0 || yIter < 0 || (xMax + 1) >= targetSize.width || yIter >= targetSize.height)
        {
            //add dummy 0 value
            addPixelToSet(0);
        }
        else
        {
            //Add if not oob.
            addPixelToSet(imagePtr->at<uchar>(yIter,(xMax + 1)));
        }
    }

    //(x,yMin--)
    //(x,yMax++)
    //(xMin--,yMin--)
    //(xMax++,yMax++)
    //(xMin--, yMax++
    //(xMax++,yMin--)
    for(int xIter = xMin - 1; xIter < (xMax + 1); xIter++)
    {
        //Detect OOB
        if((yMin - 1) < 0 || xIter < 0 || xIter >= targetSize.width)
        {
            //add dummy 0 value
            addPixelToSet(0);
        }
        else
        {
            //Add if not oob.
            addPixelToSet(imagePtr->at<uchar>((yMin - 1),xIter));
        }

        //Detect OOB
        if(xIter < 0 || (yMax + 1) >= targetSize.height || xIter >= targetSize.width)
        {
            //add dummy 0 value
            addPixelToSet(0);
        }
        else
        {
            //Add if not oob.
            addPixelToSet(imagePtr->at<uchar>((yMax + 1),xIter));
        }
    }

    xMin--;
    yMin--;
    xMax++;
    yMax++;
}

cv::Mat adaptive_median_filter(cv::Mat * targetGreyImage, int maxWindowSize)
{
    //Create object to return
    cv::Mat toReturn = targetGreyImage->clone();

    int startingWindowSize = 3;
    //Iterate through each pixel m x n
    for(unsigned int xCoord = 0; xCoord < targetGreyImage->size().width; xCoord++)
    {
        for(unsigned int yCoord = 0; yCoord < targetGreyImage->size().height; yCoord++)
        {
            MedianImageFilter filter(startingWindowSize,targetGreyImage, xCoord, yCoord);
            toReturn.at<uchar>(yCoord,xCoord) = adaptive_median_filter_stageA(filter,maxWindowSize);
        }
    }

    return toReturn;
}

int adaptive_median_filter_stageA(MedianImageFilter & adapFilter, int maxSize)
{
    int med = adapFilter.getZMed();
    if (med > adapFilter.minVal && adapFilter.maxVal > med)
    {
        return adaptive_median_filter_stageB(adapFilter,maxSize);
    }
    else
    {
        adapFilter.increaseWindowSize();
    }

    if (adapFilter.size <= maxSize)
    {
        return adaptive_median_filter_stageA(adapFilter,maxSize);
    }
    else
    {
        return med;
    }
}

int adaptive_median_filter_stageB(MedianImageFilter & adapFilter, int maxSize)
{
    if(adapFilter.centerVal > adapFilter.minVal && adapFilter.maxVal > adapFilter.centerVal)
    {
        return adapFilter.centerVal;
    }
    else
    {
        return adapFilter.getZMed();
    }
}