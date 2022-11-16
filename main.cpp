//============================================================================
// Name        : main.cpp
// Author      : Ronny Haensch
// Version     : 2.0
// Copyright   : -
// Description : only calls processing and test routines
//============================================================================


#include "Dip2.h"

#include <opencv2/opencv.hpp>

#include <stdexcept>
#include <iostream>
#include <string>
#include <sstream>




using namespace std;
using namespace cv;

cv::Mat_<float> tryLoadImage(const std::string &filename)
{
    cv::Mat img = cv::imread(filename, 0);
    if (!img.data){
        cout << "ERROR: file " << filename << " not found" << endl;
        cout << "Press enter to exit"  << endl;
        cin.get();
        exit(-3);
    }

    // convert to floating point precision
    img.convertTo(img, CV_32FC1);
    return cv::Mat_<float>(img);
}

// generates and saves different noisy versions of input image
/*
fname:   path to the input image
*/
cv::Mat_<float> generateNoisyImage(const cv::Mat_<float> &img, dip2::NoiseType noiseType)
{
    // generate images with different types of noise
    switch (noiseType) {
        case dip2::NOISE_TYPE_1: {
            // some temporary images
            Mat tmp1(img.rows, img.cols, CV_32FC1);
            Mat tmp2(img.rows, img.cols, CV_32FC1);
            // first noise operation
            float noiseLevel = 0.15;
            randu(tmp1, 0, 1);
            threshold(tmp1, tmp2, noiseLevel, 1, THRESH_BINARY);
            multiply(tmp2,img,tmp2);
            threshold(tmp1, tmp1, 1-noiseLevel, 1, THRESH_BINARY);
            tmp1 *= 255;
            tmp1 = tmp2 + tmp1;
            threshold(tmp1, tmp1, 255, 255, THRESH_TRUNC);
            return tmp1;
        } break;
        case dip2::NOISE_TYPE_2: {
            // some temporary images
            Mat tmp1(img.rows, img.cols, CV_32FC1);
            Mat tmp2(img.rows, img.cols, CV_32FC1);
            // second noise operation
            float noiseLevel = 50;
            randn(tmp1, 0, noiseLevel);
            tmp1 = img + tmp1;
            threshold(tmp1,tmp1,255,255,THRESH_TRUNC);
            threshold(tmp1,tmp1,0,0,THRESH_TOZERO);
            return tmp1;
        } break;
        default:
            throw std::runtime_error("Unhandled noise type!");
    }
}



int main(int argc, char** argv) {

   // check if enough arguments are defined
   if (argc < 2){
      cout << "Usage: ./main path_to_original_image"  << endl;
      cout << "Press enter to exit"  << endl;
      cin.get();
      return -1;
   }

    cout << "load original image" << endl;
    cv::Mat_<float> originalImage = tryLoadImage(argv[1]);
    cout << "done" << endl;

    cout << "generate noisy images" << endl;
    cv::Mat_<float> noisyImage[dip2::NUM_NOISE_TYPES];
    for (unsigned i = 0; i < dip2::NUM_NOISE_TYPES; i++) {
        noisyImage[i] = generateNoisyImage(originalImage, (dip2::NoiseType)i);
        imwrite(std::string(dip2::noiseTypeNames[i])+".jpg", noisyImage[i]);
    }
    cout << "done" << endl;


    cout << "denoising" << endl;
    cv::Mat_<float> denoisedImage[dip2::NUM_NOISE_TYPES][dip2::NUM_FILTERS];
    for (unsigned i = 0; i < dip2::NUM_NOISE_TYPES; i++)
        for (unsigned j = 0; j < dip2::NUM_FILTERS; j++) {
            denoisedImage[i][j] = denoiseImage(noisyImage[i], (dip2::NoiseType) i, (dip2::NoiseReductionAlgorithm) j);

            std::stringstream filename;
            filename << "restorated__" << dip2::noiseTypeNames[i] << "__" << dip2::noiseReductionAlgorithmNames[j] << ".jpg";
        	cv::imwrite(filename.str(), denoisedImage[i][j]);

            cv::Mat_<float> diff = denoisedImage[i][j] - originalImage;
            float meanSqrDiff = cv::mean(diff.mul(diff))[0];
            float PSNR = 10.0f * std::log10(255*255 / meanSqrDiff);

            cout << "PSNR for " << dip2::noiseTypeNames[i] << " with " << dip2::noiseReductionAlgorithmNames[j] << ": " << PSNR << " dB" << std::endl;
        }
    cout << "done (higher PSNR is better)" << endl;

    for (unsigned i = 0; i < dip2::NUM_NOISE_TYPES; i++) {
        dip2::NoiseReductionAlgorithm bestAlgorithm = chooseBestAlgorithm((dip2::NoiseType) i);

        if ((unsigned) bestAlgorithm >= dip2::NUM_FILTERS) {
            std::cout << "Error: chooseBestAlgorithm returns invalid algorithm" << std::endl;
        } else {
            std::stringstream filename;
            filename << "restorated__" << dip2::noiseTypeNames[i] << "__best.jpg";
        	cv::imwrite(filename.str(), denoisedImage[i][bestAlgorithm]);
        }
    }


	return 0;
} 
