//============================================================================
// Name        : Dip2.h
// Author      : Ronny Haensch
// Version     : 2.0
// Copyright   : -
// Description : header file for second DIP assignment
//============================================================================


#include <opencv2/opencv.hpp>

#include <iostream>

namespace dip2 {

enum NoiseType {
    NOISE_TYPE_1, /// Mysterious noise type 1 (look at the pictures, we won't tell you what it is)
    NOISE_TYPE_2, /// Mysterious noise type 2 (look at the pictures, we won't tell you what it is)
    NUM_NOISE_TYPES
};

extern const char *noiseTypeNames[NUM_NOISE_TYPES];

enum NoiseReductionAlgorithm {
    NR_MOVING_AVERAGE_FILTER,
    NR_MEDIAN_FILTER,
    NR_BILATERAL_FILTER,
    NUM_FILTERS
};

extern const char *noiseReductionAlgorithmNames[NUM_FILTERS];

// function headers of functions to be implemented
// --> please edit ONLY these functions!

/**
 * @brief Convolution in spatial domain.
 * @details Performs spatial convolution of image and filter kernel.
 * @params src Input image
 * @params kernel Filter kernel
 * @returns Convolution result
 */
cv::Mat_<float> spatialConvolution(const cv::Mat_<float>& src, const cv::Mat_<float>& kernel);

/**
 * @brief Moving average filter (aka box filter)
 * @note: you might want to use Dip2::spatialConvolution(...) within this function
 * @param src Input image
 * @param kSize Window size used by local average
 * @returns Filtered image
 */
cv::Mat_<float> averageFilter(const cv::Mat_<float>& src, int kSize);

/**
 * @brief Median filter
 * @param src Input image
 * @param kSize Window size used by median operation
 * @returns Filtered image
 */
cv::Mat_<float> medianFilter(const cv::Mat_<float>& src, int kSize);


/**
 * @brief Bilateral filer
 * @param src Input image
 * @param kSize Size of the kernel
 * @param sigma_spatial Standard-deviation of the spatial kernel
 * @param sigma_radiometric Standard-deviation of the radiometric kernel
 * @returns Filtered image
 */
cv::Mat_<float> bilateralFilter(const cv::Mat_<float>& src, int kSize, float sigma_spatial, float sigma_radiometric);

/**
 * @brief Non-local means filter
 * @note: This one is optional!
 * @param src Input image
 * @param searchSize Size of search region
 * @param sigma Optional parameter for weighting function
 * @returns Filtered image
 */
cv::Mat_<float> nlmFilter(const cv::Mat_<float>& src, int searchSize, double sigma);

/**
 * @brief Chooses the right algorithm for the given noise type
 * @note: Figure out what kind of noise NOISE_TYPE_1 and NOISE_TYPE_2 are and select the respective "right" algorithms.
 */
NoiseReductionAlgorithm chooseBestAlgorithm(NoiseType noiseType);

/**
 * @brief Denoising, with parameters specifically tweaked to the two noise types.
 * @note: Figure out reasonable denoising parameters for each algorithm-noise combination.
 */
cv::Mat_<float> denoiseImage(const cv::Mat_<float> &src, NoiseType noiseType, dip2::NoiseReductionAlgorithm noiseReductionAlgorithm);


}
