//============================================================================
// Name        : Dip4.cpp
// Author      : Ronny Haensch, Andreas Ley
// Version     : 3.0
// Copyright   : -
// Description : 
//============================================================================

#include "Dip4.h"

namespace dip4 {

using namespace std::complex_literals;

/*

===== std::complex cheat sheet ===== 

Initialization:

std::complex<float> a(1.0f, 2.0f);
std::complex<float> a = 1.0f + 2.0if;

Common Operations:

std::complex<float> a, b, c;

a = b + c;
a = b - c;
a = b * c;
a = b / c;

std::sin, std::cos, std::tan, std::sqrt, std::pow, std::exp, .... all work as expected

Access & Specific Operations:

std::complex<float> a = ...;

float real = a.real();
float imag = a.imag();
float phase = std::arg(a);
float magnitude = std::abs(a);
float squared_magnitude = std::norm(a);

std::complex<float> complex_conjugate_a = std::conj(a);

*/
    
    
    
/**
 * @brief Computes the complex valued forward DFT of a real valued input
 * @param input real valued input
 * @return Complex valued output, each pixel storing real and imaginary parts
 */
cv::Mat_<std::complex<float>> DFTReal2Complex(const cv::Mat_<float>& input)
{
    float height = input.rows;
    float width = input.cols;

    cv::Mat_<std::complex<float>> out = cv::Mat_<std::complex<float>>(height, width);
    std::complex<float> pi_c(M_PI, 0);

    // iterate over frequency domain
    for (int k=0; k<height; k++){
        for (int l=0; l<width; l++){
            std::complex<float> val = 0.0f + 0.0if;

            // iterate over spatial domain
            for (int x=0; x<height; x++){
                for (int y=0; y<width; y++){
                    std::complex<float> tmp;
                    tmp = input.at<float>(x,y) * exp(-2.0f * pi_c * 1if * ((k * x/height) + (l * y/width)));
                    val += tmp;
                    //std::cout << "val = " << std::endl << " "  << val << std::endl << std::endl;
                }
            }

            out.at<std::complex<float>>(k,l) = val;
        }
    }

    //cv::Mat_<std::complex<float>> test;
    //dft(input, test);

    //std::cout << "out = " << std::endl << " "  << out << std::endl << std::endl;
    //std::cout << "dft = " << std::endl << " "  << test << std::endl << std::endl;

    return out;
}

    
/**
 * @brief Computes the real valued inverse DFT of a complex valued input
 * @param input Complex valued input, each pixel storing real and imaginary parts
 * @return Real valued output
 */
cv::Mat_<float> IDFTComplex2Real(const cv::Mat_<std::complex<float>>& input)
{
    float height = input.rows;
    float width = input.cols;

    //std::cout << "input = " << std::endl << " "  << input << std::endl << std::endl;

    cv::Mat_<float> out = cv::Mat_<float>(height, width);
    std::complex<float> pi_c(M_PI, 0);

    // iterate over spatial domain
    for (int x=0; x<height; x++){
        for (int y=0; y<width; y++){
            std::complex<float> val = 0.0f + 0.0if;

            // iterate over frequency domain
            for (int k=0; k<height; k++){
                for (int l=0; l<width; l++){
                    //std::complex<float> tmp = input.at<std::complex<float>>(k,l);
                    val += input.at<std::complex<float>>(k,l) * exp(2.0f * pi_c * 1if * ((k * x/height) + (l * y/width)));
                    //std::cout << "val = " << std::endl << " "  << val << std::endl << std::endl;
                }
            }
            //std::cout << "out = " << std::endl << " "  << out << std::endl << std::endl;
            out.at<float>(x,y) = (val / (height * width)).real();
            //std::cout << "out = " << std::endl << " "  << out << std::endl << std::endl;
        }
    }
    //std::cout << "out = " << std::endl << " "  << out << std::endl << std::endl;
    return out;
}
    
/**
 * @brief Performes a circular shift in (dx,dy) direction
 * @param in Input matrix
 * @param dx Shift in x-direction
 * @param dy Shift in y-direction
 * @return Circular shifted matrix
*/
cv::Mat_<float> circShift(const cv::Mat_<float>& in, int dx, int dy)
{
    cv::Mat_<float> out = in.clone();

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
 * @brief Computes the thresholded inverse filter
 * @param input Blur filter in frequency domain (complex valued)
 * @param eps Factor to compute the threshold (relative to the max amplitude)
 * @return The inverse filter in frequency domain (complex valued)
 */
cv::Mat_<std::complex<float>> computeInverseFilter(const cv::Mat_<std::complex<float>>& input, const float eps)
{
    float height = input.rows;
    float width = input.cols;

    cv::Mat_<std::complex<float>> out = cv::Mat_<std::complex<float>>(height, width);

    // finding max magnitude
    float max_mag = 0;
    cv::Mat_<float> mag_input = cv::Mat_<float>(height, width);
    for (int r=0; r<height; r++){
        for (int c=0; c<width; c++){
            float abs_val = std::abs(input.at<std::complex<float>>(r, c));

            mag_input.at<float>(r,c) = abs_val;
            if (abs_val > max_mag){
                max_mag = abs_val;
            }
        }
    }

    float thresh = eps * max_mag;

    for (int r=0; r<height; r++){
        for (int c=0; c<width; c++){
            float val = mag_input.at<float>(r, c);

            if (val > thresh){
                out.at<std::complex<float>>(r, c) = (1.0f + 0if) / input.at<std::complex<float>>(r, c);
            }
            else {
                out.at<std::complex<float>>(r, c) = (1.0f + 0if) / thresh;
            }
            //std::cout << "out = " << std::endl << " "  << out << std::endl << std::endl;
             
        }
    }

    return out;
}


/**
 * @brief Applies a filter (in frequency domain)
 * @param input Image in frequency domain (complex valued)
 * @param filter Filter in frequency domain (complex valued), same size as input
 * @return The filtered image, complex valued, in frequency domain
 */
cv::Mat_<std::complex<float>> applyFilter(const cv::Mat_<std::complex<float>>& input, const cv::Mat_<std::complex<float>>& filter)
{
    //std::cout << "input = " << std::endl << " "  << input << std::endl << std::endl;
    //std::cout << "filter = " << std::endl << " "  << filter << std::endl << std::endl;
    cv::Mat_<std::complex<float>> out ;
    mulSpectrums(input, filter, out, 0);
    //std::cout << "out = " << std::endl << " "  << out << std::endl << std::endl;
    return out;
}


/**
 * @brief Function applies the inverse filter to restorate a degraded image
 * @param degraded Degraded input image
 * @param filter Filter which caused degradation
 * @param eps Factor to compute the threshold (relative to the max amplitude)
 * @return Restorated output image
 */
cv::Mat_<float> inverseFilter(const cv::Mat_<float>& degraded, const cv::Mat_<float>& filter, const float eps)
{
    int row_filter_diff = degraded.rows - filter.rows;
    int col_filter_diff = degraded.cols - filter.cols;

    cv::Mat_<float> filter_expanded;

    cv::copyMakeBorder(filter, filter_expanded, 0, row_filter_diff, 0, col_filter_diff, cv::BORDER_CONSTANT, 0);
    filter_expanded = circShift(filter_expanded, int(-filter.rows/2), int(-filter.cols/2));

    cv::Mat_<std::complex<float>> dft_degraded = DFTReal2Complex(degraded);
    cv::Mat_<std::complex<float>> dft_filter = DFTReal2Complex(filter_expanded);

    dft_filter = computeInverseFilter(dft_filter, eps);

    cv::Mat_<std::complex<float>> dft_restored = applyFilter(dft_degraded, dft_filter);

    cv::Mat_<float> restored = IDFTComplex2Real(dft_restored);

    return restored;
}


/**
 * @brief Computes the Wiener filter
 * @param input Blur filter in frequency domain (complex valued)
 * @param snr Signal to noise ratio
 * @return The wiener filter in frequency domain (complex valued)
 */
cv::Mat_<std::complex<float>> computeWienerFilter(const cv::Mat_<std::complex<float>>& input, const float snr)
{
    float height = input.rows;
    float width = input.cols;

    cv::Mat_<std::complex<float>> out = cv::Mat_<std::complex<float>>(height, width);

    for (int r=0; r<height; r++){
        for (int c=0; c<width; c++){
            std::complex<float> val = input.at<std::complex<float>>(r, c);
            std::complex<float> val_conj = std::conj(val);
            
            std::complex<float> denumerator = std::norm(val) + (1/snr);

            std::complex<float> out_val = val_conj / denumerator;
            out.at<std::complex<float>>(r,c) = out_val;    
        }
    }
    std::cout << "out = " << std::endl << " "  << out << std::endl << std::endl;
    return out;
}

/**
 * @brief Function applies the wiener filter to restore a degraded image
 * @param degraded Degraded input image
 * @param filter Filter which caused degradation
 * @param snr Signal to noise ratio of the input image
 * @return Restored output image
 */
cv::Mat_<float> wienerFilter(const cv::Mat_<float>& degraded, const cv::Mat_<float>& filter, float snr)
{
    int row_filter_diff = degraded.rows - filter.rows;
    int col_filter_diff = degraded.cols - filter.cols;

    cv::Mat_<float> filter_expanded;

    cv::copyMakeBorder(filter, filter_expanded, 0, row_filter_diff, 0, col_filter_diff, cv::BORDER_CONSTANT, 0);
    filter_expanded = circShift(filter_expanded, int(-filter.rows/2), int(-filter.cols/2));

    cv::Mat_<std::complex<float>> dft_degraded = DFTReal2Complex(degraded);
    cv::Mat_<std::complex<float>> dft_filter = DFTReal2Complex(filter_expanded);

    dft_filter = computeWienerFilter(dft_filter, snr);

    cv::Mat_<std::complex<float>> dft_restored = applyFilter(dft_degraded, dft_filter);

    cv::Mat_<float> restored = IDFTComplex2Real(dft_restored);

    return restored;
}

/* *****************************
  GIVEN FUNCTIONS
***************************** */

/**
 * function degrades the given image with gaussian blur and additive gaussian noise
 * @param img Input image
 * @param degradedImg Degraded output image
 * @param filterDev Standard deviation of kernel for gaussian blur
 * @param snr Signal to noise ratio for additive gaussian noise
 * @return The used gaussian kernel
 */
cv::Mat_<float> degradeImage(const cv::Mat_<float>& img, cv::Mat_<float>& degradedImg, float filterDev, float snr)
{

    int kSize = round(filterDev*3)*2 - 1;
   
    cv::Mat gaussKernel = cv::getGaussianKernel(kSize, filterDev, CV_32FC1);
    gaussKernel = gaussKernel * gaussKernel.t();

    cv::Mat imgs = img.clone();
    cv::dft( imgs, imgs, img.rows);
    cv::Mat kernels = cv::Mat::zeros( img.rows, img.cols, CV_32FC1);
    int dx, dy; dx = dy = (kSize-1)/2.;
    for(int i=0; i<kSize; i++) 
        for(int j=0; j<kSize; j++) 
            kernels.at<float>((i - dy + img.rows) % img.rows,(j - dx + img.cols) % img.cols) = gaussKernel.at<float>(i,j);
	cv::dft( kernels, kernels );
	cv::mulSpectrums( imgs, kernels, imgs, 0 );
	cv::dft( imgs, degradedImg,  cv::DFT_INVERSE + cv::DFT_SCALE, img.rows );
	
    cv::Mat mean, stddev;
    cv::meanStdDev(img, mean, stddev);

    cv::Mat noise = cv::Mat::zeros(img.rows, img.cols, CV_32FC1);
    cv::randn(noise, 0, stddev.at<double>(0)/snr);
    degradedImg = degradedImg + noise;
    cv::threshold(degradedImg, degradedImg, 255, 255, cv::THRESH_TRUNC);
    cv::threshold(degradedImg, degradedImg, 0, 0, cv::THRESH_TOZERO);

    return gaussKernel;
}


}
