//============================================================================
// Name    : Dip3.cpp
// Author      : Ronny Haensch, Andreas Ley
// Version     : 3.0
// Copyright   : -
// Description : 
//============================================================================

#include "Dip3.h"

#include <stdexcept>

namespace dip3 {

const char * const filterModeNames[NUM_FILTER_MODES] = {
    "FM_SPATIAL_CONVOLUTION",
    "FM_FREQUENCY_CONVOLUTION",
    "FM_SEPERABLE_FILTER",
    "FM_INTEGRAL_IMAGE",
};



/**
 * @brief Generates 1D gaussian filter kernel of given size
 * @param kSize Kernel size (used to calculate standard deviation)
 * @returns The generated filter kernel
 */
cv::Mat_<float> createGaussianKernel1D(int kSize){

    // TO DO !!!
   
    return cv::Mat_<float>::zeros(1, kSize);
}

/**
 * @brief Generates 2D gaussian filter kernel of given size
 * @param kSize Kernel size (used to calculate standard deviation)
 * @returns The generated filter kernel
 */
cv::Mat_<float> createGaussianKernel2D(int kSize){

    // TO DO !!!
   
    return cv::Mat_<float>::zeros(kSize, kSize);
}

/**
 * @brief Performes a circular shift in (dx,dy) direction
 * @param in Input matrix
 * @param dx Shift in x-direction
 * @param dy Shift in y-direction
 * @returns Circular shifted matrix
 */
cv::Mat_<float> circShift(const cv::Mat_<float>& in, int dx, int dy){

   // TO DO !!!

   return in;
}


/**
 * @brief Performes convolution by multiplication in frequency domain
 * @param in Input image
 * @param kernel Filter kernel
 * @returns Output image
 */
cv::Mat_<float> frequencyConvolution(const cv::Mat_<float>& in, const cv::Mat_<float>& kernel){

   // TO DO !!!

   return in;
}


/**
 * @brief  Performs UnSharp Masking to enhance fine image structures
 * @param in The input image
 * @param filterMode How convolution for smoothing operation is done
 * @param size Size of used smoothing kernel
 * @param thresh Minimal intensity difference to perform operation
 * @param scale Scaling of edge enhancement
 * @returns Enhanced image
 */
cv::Mat_<float> usm(const cv::Mat_<float>& in, FilterMode filterMode, int size, float thresh, float scale)
{
   // TO DO !!!

   // use smoothImage(...) for smoothing

   return in;
}


/**
 * @brief Convolution in spatial domain
 * @param src Input image
 * @param kernel Filter kernel
 * @returns Convolution result
 */
cv::Mat_<float> spatialConvolution(const cv::Mat_<float>& src, const cv::Mat_<float>& kernel)
{

   // Hopefully already DONE, copy from last homework

   return src;
}


/**
 * @brief Convolution in spatial domain by seperable filters
 * @param src Input image
 * @param size Size of filter kernel
 * @returns Convolution result
 */
cv::Mat_<float> separableFilter(const cv::Mat_<float>& src, const cv::Mat_<float>& kernel){

   // TO DO !!!

   return src;

}


/**
 * @brief Convolution in spatial domain by integral images
 * @param src Input image
 * @param size Size of filter kernel
 * @returns Convolution result
 */
cv::Mat_<float> satFilter(const cv::Mat_<float>& src, int size){

   // optional

   return src;

}

/* *****************************
  GIVEN FUNCTIONS
***************************** */

/**
 * @brief Performs a smoothing operation but allows the algorithm to be chosen
 * @param in Input image
 * @param size Size of filter kernel
 * @param type How is smoothing performed?
 * @returns Smoothed image
 */
cv::Mat_<float> smoothImage(const cv::Mat_<float>& in, int size, FilterMode filterMode)
{
    switch(filterMode) {
        case FM_SPATIAL_CONVOLUTION: return spatialConvolution(in, createGaussianKernel2D(size));	// 2D spatial convolution
        case FM_FREQUENCY_CONVOLUTION: return frequencyConvolution(in, createGaussianKernel2D(size));	// 2D convolution via multiplication in frequency domain
        case FM_SEPERABLE_FILTER: return separableFilter(in, createGaussianKernel1D(size));	// seperable filter
        case FM_INTEGRAL_IMAGE: return satFilter(in, size);		// integral image
        default: 
            throw std::runtime_error("Unhandled filter type!");
    }
}



}

