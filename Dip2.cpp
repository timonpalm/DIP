//============================================================================
// Name        : Dip2.cpp
// Author      : Ronny Haensch
// Version     : 2.0
// Copyright   : -
// Description : 
//============================================================================

#include "Dip2.h"

namespace dip2 {


/**
 * @brief Convolution in spatial domain.
 * @details Performs spatial convolution of image and filter kernel.
 * @params src Input image
 * @params kernel Filter kernel
 * @returns Convolution result
 */
cv::Mat_<float> spatialConvolution(const cv::Mat_<float>& src, const cv::Mat_<float>& kernel)
{
    //std::cout << "src = " << std::endl << " "  << src << std::endl << std::endl;
    cv::Mat_<float> conv_src;

    float kernel_size = kernel.rows; // assuming kernel is quadratic and odd numbered
    int kernel_midpoint = kernel_size / 2;
    
    cv::copyMakeBorder( src, conv_src, kernel_midpoint, kernel_midpoint, kernel_midpoint, kernel_midpoint, cv::BORDER_CONSTANT, 1);
    cv::Mat_<float> output = src.clone();

    //std::cout << "conv_src = " << std::endl << " "  << conv_src << std::endl << std::endl;

    cv::Mat kernel_flip;
    cv::flip(kernel, kernel_flip, 0);
    cv::Mat kernel_flat = kernel_flip.reshape(1,1); // flatten kernel to 1d vector
    //std::cout << "kernel = " << std::endl << " "  << kernel << std::endl << std::endl;
    //std::cout << "kernel_flip = " << std::endl << " "  << kernel_flip << std::endl << std::endl;

    //std::cout << kernel.convertTo << std::endl;

    for(int row=kernel_midpoint; row<conv_src.rows-kernel_midpoint-1; row++)
    {
        for(int col=kernel_midpoint; col<conv_src.cols-kernel_midpoint-1; col++)
        {
            cv::Rect r(col-kernel_midpoint, row-kernel_midpoint, kernel_size, kernel_size);
            cv::Mat pixels = conv_src(r).clone();
            //std::cout << "pixels = " << std::endl << " "  << pixels << std::endl << std::endl;
            pixels = pixels.reshape(1,1);

            float new_val = kernel_flat.dot(pixels);
            //std::cout << "o = " << std::endl << " "  << output << std::endl << std::endl;
            output.at<float>(row-kernel_midpoint, col-kernel_midpoint) = new_val;
            
        }
    }
        
    //std::cout << "ouput = " << std::endl << " "  << output << std::endl << std::endl;

    return output;
}

/**
 * @brief Moving average filter (aka box filter)
 * @note: you might want to use Dip2::spatialConvolution(...) within this function
 * @param src Input image
 * @param kSize Window size used by local average
 * @returns Filtered image
 */
cv::Mat_<float> averageFilter(const cv::Mat_<float>& src, int kSize)
{
    float val = 1.0 / (kSize * kSize);
    cv::Mat kernel(cv::Size(kSize, kSize), CV_32FC1, cv::Scalar(val));
    //std::cout << "kernel = " << std::endl << " "  << kernel << std::endl << std::endl;
    cv::Mat output = spatialConvolution(src, kernel);
    return output;
}

/**
 * @brief Median filter
 * @param src Input image
 * @param kSize Window size used by median operation
 * @returns Filtered image
 */
cv::Mat_<float> medianFilter(const cv::Mat_<float>& src, int kSize)
{
    //std::cout << "src = " << std::endl << " "  << src << std::endl << std::endl;

    cv::Mat_<float> src_b;
    cv::copyMakeBorder( src, src_b, 1, 1, 1, 1, cv::BORDER_CONSTANT, 0);
    cv::Mat_<float> output = src.clone();

    //std::cout << "conv_src = " << std::endl << " "  << conv_src << std::endl << std::endl;

    int median_idx = (kSize*kSize) / 2;
    int kernel_midpoint = kSize / 2;

    for(int row=1; row<src_b.rows-1; row++)
    {
        for(int col=1; col<src_b.cols-1; col++)
        {
            cv::Rect r(col-kernel_midpoint, row-kernel_midpoint, kSize, kSize);
            cv::Mat pixels = src_b(r).clone();
            
            cv::sort(pixels, pixels, cv::SORT_EVERY_ROW);
            

            float median = pixels.at<float>(0, median_idx);
            output.at<float>(row-1, col-1) = median;
            
        }
    }
        
    //std::cout << "ouput = " << std::endl << " "  << output << std::endl << std::endl;

    return output;
}

/**
 * @brief Bilateral filer
 * @param src Input image
 * @param kSize Size of the kernel
 * @param sigma_spatial Standard-deviation of the spatial kernel
 * @param sigma_radiometric Standard-deviation of the radiometric kernel
 * @returns Filtered image
 */
cv::Mat_<float> bilateralFilter(const cv::Mat_<float>& src, int kSize, float sigma_spatial, float sigma_radiometric)
{

    // tagret pixel is at 0,0 so we start at the upper left, e.g. -1,-1 depending on the kernel size
    int kernel_midpoint = kSize / 2;

    cv::Mat conv_src;
    cv::copyMakeBorder( src, conv_src, kernel_midpoint, kernel_midpoint, kernel_midpoint, kernel_midpoint, cv::BORDER_CONSTANT, 1);
    cv::Mat_<float> output = src.clone();

    for(int row=kernel_midpoint; row<conv_src.rows-kernel_midpoint-1; row++)
    {
        for(int col=kernel_midpoint; col<conv_src.cols-kernel_midpoint-1; col++)
        {
            cv::Rect r(col-kernel_midpoint, row-kernel_midpoint, kSize, kSize);

            float val_midpoint = conv_src.at<float>(row, col);

            float w_sum = 0;
            float val_sum = 0;
            for(int x=-kernel_midpoint; x<=kernel_midpoint; x++)
            {
                for(int y=-kernel_midpoint; y<=kernel_midpoint; y++)
                {
                    float h_spat = (1 / (2 * M_PI * pow(sigma_spatial, 2))) * exp( (- (pow(x, 2) + pow(y, 2)))/ (2 * pow(sigma_spatial, 2)));

                    float kernel_val = conv_src.at<float>(row+x, col+y);

                    float h_radio = (1 / (2 * M_PI * pow(sigma_radiometric, 2))) * exp( -pow(kernel_val - val_midpoint, 2)/ (2 * pow(sigma_radiometric, 2)));

                    float w = h_spat * h_radio;
                    float val = w * kernel_val;

                    w_sum += w;
                    val_sum += val;
                    
                }
            }

            output.at<float>(row-kernel_midpoint, col-kernel_midpoint) = val_sum / w_sum;
            
        }
    }
    return output;
}

/**
 * @brief Non-local means filter
 * @note: This one is optional!
 * @param src Input image
 * @param searchSize Size of search region
 * @param sigma Optional parameter for weighting function
 * @returns Filtered image
 */
cv::Mat_<float> nlmFilter(const cv::Mat_<float>& src, int searchSize, double sigma)
{
    return src.clone();
}



/**
 * @brief Chooses the right algorithm for the given noise type
 * @note: Figure out what kind of noise NOISE_TYPE_1 and NOISE_TYPE_2 are and select the respective "right" algorithms.
 */
NoiseReductionAlgorithm chooseBestAlgorithm(NoiseType noiseType)
{
    // TO DO !!
    return (NoiseReductionAlgorithm) -1;
}



cv::Mat_<float> denoiseImage(const cv::Mat_<float> &src, NoiseType noiseType, dip2::NoiseReductionAlgorithm noiseReductionAlgorithm)
{
    // TO DO !!

    // for each combination find reasonable filter parameters

    switch (noiseReductionAlgorithm) {
        case dip2::NR_MOVING_AVERAGE_FILTER:
            switch (noiseType) {
                case NOISE_TYPE_1:
                    return dip2::averageFilter(src, 1);
                case NOISE_TYPE_2:
                    return dip2::averageFilter(src, 1);
                default:
                    throw std::runtime_error("Unhandled noise type!");
            }
        case dip2::NR_MEDIAN_FILTER:
            switch (noiseType) {
                case NOISE_TYPE_1:
                    return dip2::medianFilter(src, 1);
                case NOISE_TYPE_2:
                    return dip2::medianFilter(src, 1);
                default:
                    throw std::runtime_error("Unhandled noise type!");
            }
        case dip2::NR_BILATERAL_FILTER:
            switch (noiseType) {
                case NOISE_TYPE_1:
                    return dip2::bilateralFilter(src, 1, 1.0f, 1.0f);
                case NOISE_TYPE_2:
                    return dip2::bilateralFilter(src, 1, 1.0f, 1.0f);
                default:
                    throw std::runtime_error("Unhandled noise type!");
            }
        default:
            throw std::runtime_error("Unhandled filter type!");
    }
}





// Helpers, don't mind these

const char *noiseTypeNames[NUM_NOISE_TYPES] = {
    "NOISE_TYPE_1",
    "NOISE_TYPE_2",
};

const char *noiseReductionAlgorithmNames[NUM_FILTERS] = {
    "NR_MOVING_AVERAGE_FILTER",
    "NR_MEDIAN_FILTER",
    "NR_BILATERAL_FILTER",
};


}
