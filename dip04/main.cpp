//============================================================================
// Name        : main.cpp
// Author      : Ronny Haensch
// Version     : 2.0
// Copyright   : -
// Description : only calls processing and test routines
//============================================================================

#include <iostream>

#include "Dip4.h"

using namespace std;
using namespace cv;

/**
 * @brief Function displays image (after proper normalization)
 * @param win Window name
 * @param img Image that shall be displayed
 * @param cut Determines whether to cut or scale values outside of [0,255] range
 */
void showImage(const char* win, const cv::Mat_<float>& img, bool cut = true)
{
    cv::Mat tmp = img.clone();

    if (tmp.channels() == 1) {
        if (cut){
            threshold(tmp, tmp, 255, 255, THRESH_TRUNC);
            threshold(tmp, tmp, 0, 0, THRESH_TOZERO);
        }else
            normalize(tmp, tmp, 0, 255, cv::NORM_MINMAX);

        tmp.convertTo(tmp, CV_8UC1);
    }else{
        tmp.convertTo(tmp, CV_8UC3);
    }
    imshow(win, tmp);
}




// usage: path to image in argv[1], SNR in argv[2], stddev of Gaussian blur in argv[3]
// main function. Loads the image, calls test and processing routines, records processing times
int main(int argc, char** argv) {

   // check if enough arguments are defined
   if (argc < 4){
      cout << "Usage:\n\tdip4 path_to_original snr stddev"  << endl;
      cout << "\t\t snr :\t\tsignal-to-noise ratio: the higher (e.g. 10,000), the less noise." << endl;
      cout << "\t\t stddev :\tstddev of Gaussian blur" << endl;
      cout << "Press enter to exit"  << endl;
      cin.get();
      return -1;
   }

	// some windows for displaying imagesvoid degradeImage(Mat imgIn32F, Mat degradedImg, double filterDev, double snr)
    const char* win_1 = "Original Image";
    const char* win_2 = "Degraded Image";
    const char* win_3 = "Restored Image: Inverse filter";
    const char* win_4 = "Restored Image: Wiener filter";
    namedWindow( win_1 );
    namedWindow( win_2 );
    namedWindow( win_3 );
    namedWindow( win_4 );
   
    // load image, path in argv[1]
    cout << "load image" << endl;
    Mat img = imread(argv[1], 0);
    if (!img.data){
      cout << "ERROR: Cannot find original image"  << endl;
      cout << "Press enter to exit..."  << endl;
      cin.get();
      return -1;
    }
    // convert U8 to 32F
    img.convertTo(img, CV_32FC1);
    cout << " > done" << endl;

    // show and safe gray-scale version of original image
    showImage( win_1, img);
    imwrite( "original.png", img );
  
    // degrade image
    cout << "degrade image" << endl;
    double snr = atof(argv[2]);
    double filterDev = atof(argv[3]);
    Mat_<float> degradedImg;
    Mat_<float> gaussKernel = dip4::degradeImage(img, degradedImg, filterDev, snr);
    cout << " > done" << endl;
    
    // show and safe degraded image
    showImage( win_2, degradedImg);
    imwrite( "degraded.png", degradedImg );
   
    // inverse filter
    cout << "inverse filter" << endl;
    Mat restoredImgInverseFilter = dip4::inverseFilter(degradedImg, gaussKernel);
    cout << " > done" << endl;
    // show and safe restored image
    showImage( win_3, restoredImgInverseFilter);
    imwrite( "restored_inverse.png", restoredImgInverseFilter );
    
    // wiener filter
    cout << "wiener filter" << endl;
    Mat restoredImgWienerFilter = dip4::wienerFilter(degradedImg, gaussKernel, snr);
    cout << " > done" << endl;
    // show and safe restored image
    showImage( win_4, restoredImgWienerFilter, false);
    imwrite( "restored_wiener.png", restoredImgWienerFilter );

    // wait
    waitKey(0);

    return 0;
} 
