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

    float varience = kSize / 5.0;

    cv::Mat_<float> kernel = cv::Mat_<float>::zeros(1, kSize);

    int midpoint = int(kSize / 2);

    float sum = 0;
    for(int i=-midpoint; i<=midpoint; i++){
      float val = (1 / (2 * M_PI * varience)) * exp(-0.5 * ((i*i) / (varience*varience)));
      kernel.at<float>(0,midpoint + i) = val;
      sum += val;
    }
   
    kernel = kernel / sum;
   
    return kernel;
}

/**
 * @brief Generates 2D gaussian filter kernel of given size
 * @param kSize Kernel size (used to calculate standard deviation)
 * @returns The generated filter kernel
 */
cv::Mat_<float> createGaussianKernel2D(int kSize){

    float varience = kSize / 5.0;

    cv::Mat_<float> kernel = cv::Mat_<float>::zeros(kSize, kSize);

    int midpoint = int(kSize / 2);

    float sum = 0;
    for(int i=-midpoint; i<=midpoint; i++){
      for(int k=-midpoint; k<=midpoint; k++){
         float val = (1 / (2 * M_PI * varience * varience)) * exp(-0.5 * (((i*i) + (k*k))/ (varience*varience)));
         kernel.at<float>(midpoint + k,midpoint + i) = val;
         sum += val;
      }
    }
   
    kernel = kernel / sum;
   
    return kernel;
}

/**
 * @brief Performes a circular shift in (dx,dy) direction
 * @param in Input matrix
 * @param dx Shift in x-direction
 * @param dy Shift in y-direction
 * @returns Circular shifted matrix
 */
cv::Mat_<float> circShift(const cv::Mat_<float>& in, int dx, int dy){

   cv::Mat_<float> out = in.clone();
   //std::cout << "dx = " << dx << std::endl;
   //std::cout << "dy = " << dy << std::endl;

   //cv::copyMakeBorder( in, out, abs(dy), abs(dy), abs(dx), abs(dx), cv::BORDER_CONSTANT, 1);

   for(int row=0; row<out.rows; row++){
      for(int col=0; col<out.cols; col++){
         int new_x = col + dx;
         int new_y = row + dy;

         if(new_x < 0){
            new_x = out.rows + new_x;
         }
         else if(new_x >= out.rows){
            new_x = new_x - out.rows;
         }
         if(new_y < 0){
            new_y = out.cols + new_y;
         }
         else if(new_y >= out.cols){
            new_y = new_y - out.cols;
         }

         out.at<float>(new_y, new_x) = in.at<float>(row, col);
         //std::cout << "out = " << std::endl << " "  << out << std::endl << std::endl;
      }
   }

   return out;
}


/**
 * @brief Performes convolution by multiplication in frequency domain
 * @param in Input image
 * @param kernel Filter kernel
 * @returns Output image
 */
cv::Mat_<float> frequencyConvolution(const cv::Mat_<float>& in, const cv::Mat_<float>& kernel){

   cv::Mat_<float> in_dft;
   cv::Mat_<float> kernel_dft;

   cv::Mat_<float> out_dft;

   int row_in_diff = cv::getOptimalDFTSize(in.rows) - in.rows;
   int col_in_diff = cv::getOptimalDFTSize(in.cols) - in.cols;

   int row_kernel_diff = (int) (cv::getOptimalDFTSize(in.rows) - kernel.rows) / 2;
   int col_kernel_diff = (int) (cv::getOptimalDFTSize(in.cols) - kernel.cols) / 2;

   cv::Mat_<float> kernel_expanded;

   cv::copyMakeBorder(in, in_dft, 0, row_in_diff, 0, col_in_diff, cv::BORDER_CONSTANT, 0);
   cv::copyMakeBorder(kernel, kernel_expanded, row_kernel_diff, row_kernel_diff, col_kernel_diff, col_kernel_diff, cv::BORDER_CONSTANT, 0);

   dft(in_dft, in_dft, 0);
   
   cv::Mat_<float> kernel_shifted = circShift(kernel_expanded, int(-kernel_expanded.rows/2), int(-kernel_expanded.cols/2));
   dft(kernel_shifted, kernel_dft, 0);
   
   mulSpectrums(in_dft, kernel_dft, out_dft, 0);

   cv::Mat_<float> out;

   dft( out_dft, out, cv::DFT_INVERSE + cv::DFT_SCALE);

   //std::cout << "out = " << std::endl << " "  << out << std::endl << std::endl;

   return out;
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

    for(int row=kernel_midpoint; row<conv_src.rows-kernel_midpoint; row++)
    {
        for(int col=kernel_midpoint; col<conv_src.cols-kernel_midpoint; col++)
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

