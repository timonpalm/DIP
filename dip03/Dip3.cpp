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
    //"FM_INTEGRAL_IMAGE",
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

   std::cout << "osize = " << std::endl << " "  << (cv::getOptimalDFTSize(in.rows) - kernel.rows) << std::endl << std::endl;
   int row_in_diff = cv::getOptimalDFTSize(in.rows) - in.rows;
   int col_in_diff = cv::getOptimalDFTSize(in.cols) - in.cols;

   int row_kernel_diff = cv::getOptimalDFTSize(in.rows) - kernel.rows;
   int col_kernel_diff = cv::getOptimalDFTSize(in.cols) - kernel.cols;

   cv::Mat_<float> kernel_expanded;

   cv::copyMakeBorder(in, in_dft, 0, row_in_diff, 0, col_in_diff, cv::BORDER_CONSTANT, 0);
   cv::copyMakeBorder(kernel, kernel_expanded, 0, row_kernel_diff, 0, col_kernel_diff, cv::BORDER_CONSTANT, 0);

   dft(in_dft, in_dft, 0);
   
   cv::Mat_<float> kernel_shifted = circShift(kernel_expanded, int(-kernel.rows/2), int(-kernel.cols/2));
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
   cv::Mat_<float> img_smooth = smoothImage(in, size, filterMode);

   cv::Mat_<float> diff;
   std::cout << "in = " << std::endl << " "  << in.at<float>(0,0) << std::endl << std::endl;
   std::cout << "smooth = " << std::endl << " "  << img_smooth.at<float>(0,0) << std::endl << std::endl;
   subtract(in, img_smooth, diff);
   std::cout << "diff = " << std::endl << " "  << diff.at<float>(0,0) << std::endl << std::endl;

   cv::Mat_<float> diff_greater;
   cv::Mat_<float> diff_smaller;
   threshold(diff, diff_greater, thresh, scale, cv::THRESH_TOZERO);
   threshold(diff, diff_smaller, -thresh, scale, cv::THRESH_TOZERO_INV);
   diff = diff_greater + diff_smaller;
   std::cout << "diff = " << std::endl << " "  << diff.at<float>(0,0) << std::endl << std::endl;
   //std::cout << "diff = " << std::endl << " "  << diff << std::endl << std::endl;

   diff = diff * scale;
   std::cout << "diff = " << std::endl << " "  << diff.at<float>(0,0) << std::endl << std::endl;
   //std::cout << "diff = " << std::endl << " "  << diff << std::endl << std::endl;

   diff = diff + in;
   std::cout << "diff = " << std::endl << " "  << diff.at<float>(0,0) << std::endl << std::endl;
   //std::cout << "diff = " << std::endl << " "  << diff << std::endl << std::endl;

   return diff;
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

    int kernel_mid_row = kernel.rows / 2;
    int kernel_mid_col = kernel.cols / 2;

    //std::cout << "kernel = " << std::endl << " "  << kernel << std::endl << std::endl;
    
    cv::copyMakeBorder( src, conv_src, kernel_mid_row, kernel_mid_row, kernel_mid_col, kernel_mid_col, cv::BORDER_CONSTANT, 1);
    cv::Mat_<float> output = src.clone();

    //std::cout << "conv_src = " << std::endl << " "  << conv_src << std::endl << std::endl;

    cv::Mat kernel_flip;
    cv::flip(kernel, kernel_flip, 0);
    cv::Mat kernel_flat = kernel_flip.reshape(1,1); // flatten kernel to 1d vector
    //std::cout << "kernel = " << std::endl << " "  << kernel << std::endl << std::endl;
    //std::cout << "kernel_flip = " << std::endl << " "  << kernel_flip << std::endl << std::endl;

    //std::cout << kernel.convertTo << std::endl;

    for(int row=kernel_mid_row; row<conv_src.rows-kernel_mid_row; row++)
    {
        for(int col=kernel_mid_col; col<conv_src.cols-kernel_mid_col; col++)
        {
            cv::Rect r(col-kernel_mid_col, row-kernel_mid_row, kernel.cols, kernel.rows);
            cv::Mat pixels = conv_src(r).clone();
            //std::cout << "pixels = " << std::endl << " "  << pixels << std::endl << std::endl;
            pixels = pixels.reshape(1,1);

            float new_val = kernel_flat.dot(pixels);
            //std::cout << "o = " << std::endl << " "  << output << std::endl << std::endl;
            output.at<float>(row-kernel_mid_row, col-kernel_mid_col) = new_val;
            
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

   cv::Mat_<float> tmp = spatialConvolution(src, kernel);
   transpose(tmp, tmp);
   cv::Mat_<float> out = spatialConvolution(tmp, kernel);

   transpose(out, out);
   return out;

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
        //case FM_INTEGRAL_IMAGE: return satFilter(in, size);		// integral image
        default: 
            throw std::runtime_error("Unhandled filter type!");
    }
}



}

