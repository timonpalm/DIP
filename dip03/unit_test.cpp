//============================================================================
// Name        : unit_test.cpp
// Author      : Ronny Haensch, Andreas Ley
// Version     : 3.0
// Copyright   : -
// Description : only calls processing and test routines
//============================================================================


#include "Dip3.h"

#include <opencv2/opencv.hpp>

#include <iostream>

using namespace std;
using namespace cv;
using namespace dip3;

union FloatInt {
    uint32_t i;
    float f;
};

inline bool fastmathIsFinite(float f)
{
    FloatInt f2i;
    f2i.f = f;
    return ((f2i.i >> 23) & 0xFF) != 0xFF;
}


bool matrixIsFinite(const Mat_<float> &mat) {
    
    for (unsigned r = 0; r < mat.rows; r++)
        for (unsigned c = 0; c < mat.cols; c++)
            if (!fastmathIsFinite(mat(r, c)))
                return false;
    
    return true;
}

bool test_createGaussianKernel1D(void)
{
   Mat k = createGaussianKernel1D(11);

   if (k.rows != 1){
      cout << "ERROR: Dip3::createGaussianKernel1D(): Wrong size!" << endl;
      return false;
   }
   if (k.cols != 11){
      cout << "ERROR: Dip3::createGaussianKernel1D(): Wrong size!" << endl;
      return false;
   }

    if (!matrixIsFinite(k)){
        cout << "ERROR: Dip3::createGaussianKernel1D(): Inf/nan values in result!" << endl;
        return false;
    }
   
   if ( abs(sum(k).val[0] - 1) > 0.0001){
      cout << "ERROR: Dip3::createGaussianKernel1D(): Sum of all kernel elements is not one!" << endl;
      return false;
   }
   if (sum(k >= k.at<float>(0,5)).val[0]/255 != 1){
      cout << "ERROR: Dip3::createGaussianKernel1D(): Seems like kernel is not centered!" << endl;
      return false;
   }
   cout << "Message: Dip3::createGaussianKernel1D() seems to be correct" << endl;
    return true;
}

bool test_createGaussianKernel2D(void)
{
   Mat k = createGaussianKernel2D(11);
   
   if (k.rows != 11){
      cout << "ERROR: Dip3::createGaussianKernel1D(): Wrong size!" << endl;
      return false;
   }
   if (k.cols != 11){
      cout << "ERROR: Dip3::createGaussianKernel1D(): Wrong size!" << endl;
      return false;
   }

    if (!matrixIsFinite(k)){
        cout << "ERROR: Dip3::createGaussianKernel2D(): Inf/nan values in result!" << endl;
        return false;
    }

   if ( abs(sum(k).val[0] - 1) > 0.0001){
      cout << "ERROR: Dip3::test_createGaussianKernel2D(): Sum of all kernel elements is not one!" << endl;
      return false;
   }
   if (sum(k >= k.at<float>(5,5)).val[0]/255 != 1){
      cout << "ERROR: Dip3::test_createGaussianKernel2D(): Seems like kernel is not centered!" << endl;
      return false;
   }
   cout << "Message: Dip3::test_createGaussianKernel2D() seems to be correct" << endl;
    return true;
}

bool test_circShift(void)
{   
    {
        Mat_<float> in(3,3);
        in.setTo(0.0f);
        in.at<float>(0,0) = 1;
        in.at<float>(0,1) = 2;
        in.at<float>(1,0) = 3;
        in.at<float>(1,1) = 4;
        Mat_<float> ref(3,3);
        ref.setTo(0.0f);
        ref.at<float>(0,0) = 4;
        ref.at<float>(0,2) = 3;
        ref.at<float>(2,0) = 2;
        ref.at<float>(2,2) = 1;
        
        Mat_<float> res = circShift(in, -1, -1);
        if (!matrixIsFinite(res)){
            cout << "ERROR: Dip3::circShift(): Inf/nan values in result!" << endl;
            return false;
        }

        if (sum((res == ref)).val[0]/255 != 9){
            cout << "ERROR: Dip3::circShift(): Result of circshift seems to be wrong!" << endl;
            return false;
        }
    }
    {
        cv::Mat_<float> in(30, 30);
        cv::randn(in, cv::Scalar(0.0f), cv::Scalar(1.0f));
        
        cv::Mat_<float> tmp;
        tmp = circShift(in, -5, -10);
        tmp = circShift(tmp, 10, -10);
        tmp = circShift(tmp, -5, 20);

        if (!matrixIsFinite(tmp)){
            cout << "ERROR: Dip3::circShift(): Inf/nan values in result!" << endl;
            return false;
        }

        if (sum(tmp != in).val[0] != 0){
            cout << "ERROR: Dip3::circShift(): Result of circshift seems to be wrong!" << endl;
            return false;
        }
    }
    return true;
}

bool test_frequencyConvolution(void)
{   
   Mat input = Mat::ones(9,9, CV_32FC1);
   input.at<float>(4,4) = 255;
   Mat kernel = Mat(3,3, CV_32FC1, 1./9.);

   Mat_<float> output = frequencyConvolution(input, kernel);
   
    if (!matrixIsFinite(output)){
        cout << "ERROR: Dip3::frequencyConvolution(): Inf/nan values in result!" << endl;
        return false;
    }


   if ( (sum(output < 0).val[0] > 0) or (sum(output > 255).val[0] > 0) ){
      cout << "ERROR: Dip3::frequencyConvolution(): Convolution result contains too large/small values!" << endl;
      return false;
   }
   float ref[9][9] = {{0, 0, 0, 0, 0, 0, 0, 0, 0},
                      {0, 1, 1, 1, 1, 1, 1, 1, 0},
                      {0, 1, 1, 1, 1, 1, 1, 1, 0},
                      {0, 1, 1, (8+255)/9., (8+255)/9., (8+255)/9., 1, 1, 0},
                      {0, 1, 1, (8+255)/9., (8+255)/9., (8+255)/9., 1, 1, 0},
                      {0, 1, 1, (8+255)/9., (8+255)/9., (8+255)/9., 1, 1, 0},
                      {0, 1, 1, 1, 1, 1, 1, 1, 0},
                      {0, 1, 1, 1, 1, 1, 1, 1, 0},
                      {0, 0, 0, 0, 0, 0, 0, 0, 0}};
   for(int y=1; y<8; y++){
      for(int x=1; x<8; x++){
         if (abs(output.at<float>(y,x) - ref[y][x]) > 0.0001){
            cout << "ERROR: Dip3::frequencyConvolution(): Convolution result contains wrong values!" << endl;
            return false;
         }
      }
   }
   cout << "Message: Dip3::frequencyConvolution() seems to be correct" << endl;
    return true;
}

bool test_separableConvolution(void)
{   
   Mat input = Mat::ones(9,9, CV_32FC1);
   input.at<float>(4,4) = 255;
   Mat kernel = Mat(1,3, CV_32FC1, 1./3.);

   Mat_<float> output = separableFilter(input, kernel);
   
    if (!matrixIsFinite(output)){
        cout << "ERROR: Dip3::separableConvolution(): Inf/nan values in result!" << endl;
        return false;
    }


   if ( (sum(output < 0).val[0] > 0) or (sum(output > 255).val[0] > 0) ){
      cout << "ERROR: Dip3::separableFilter(): Convolution result contains too large/small values!" << endl;
      return false;
   }
   float ref[9][9] = {{0, 0, 0, 0, 0, 0, 0, 0, 0},
                      {0, 1, 1, 1, 1, 1, 1, 1, 0},
                      {0, 1, 1, 1, 1, 1, 1, 1, 0},
                      {0, 1, 1, (8+255)/9., (8+255)/9., (8+255)/9., 1, 1, 0},
                      {0, 1, 1, (8+255)/9., (8+255)/9., (8+255)/9., 1, 1, 0},
                      {0, 1, 1, (8+255)/9., (8+255)/9., (8+255)/9., 1, 1, 0},
                      {0, 1, 1, 1, 1, 1, 1, 1, 0},
                      {0, 1, 1, 1, 1, 1, 1, 1, 0},
                      {0, 0, 0, 0, 0, 0, 0, 0, 0}};

   for(int y=1; y<8; y++){
      for(int x=1; x<8; x++){
         if (abs(output.at<float>(y,x) - ref[y][x]) > 0.0001){
            cout << "ERROR: Dip3::separableFilter(): Convolution result contains wrong values!" << endl;
            return false;
         }
      }
   }
   cout << "Message: Dip3::separableFilter() seems to be correct" << endl;
    return true;
}


int main(int argc, char** argv) {

    bool ok = true;

    ok &= test_createGaussianKernel1D();
    ok &= test_createGaussianKernel2D();
    ok &= test_circShift();
    ok &= test_frequencyConvolution();
    ok &= test_separableConvolution();

    if (!ok)
        return -1;
    else
    	return 0;
} 
