//============================================================================
// Name        : main.cpp
// Author      : Ronny Haensch, Andreas Ley
// Version     : 3.0
// Copyright   : -
// Description : only calls processing and test routines
//============================================================================

#include "Dip3.h"


#include <iostream>
#include <fstream>
#include <sstream>

#include <chrono>


class StopWatch {
    public:
        StopWatch() {
            m_startTime = std::chrono::high_resolution_clock::now();
        }

        float getElapsedSeconds() {
            auto time = std::chrono::duration<float>(std::chrono::high_resolution_clock::now() - m_startTime);
            return time.count();
        }
    protected:
        std::chrono::high_resolution_clock::time_point m_startTime;
};



cv::Mat_<cv::Vec3b> processColorImage(const cv::Mat_<cv::Vec3b> &src, dip3::FilterMode filterMode, int size, float thresh, float scale)
{
    cv::Mat img;
    // convert and split input image
    // convert U8 to 32F
    src.convertTo(img, CV_32FC3);
    // convert BGR to HSV
    cv::cvtColor(img, img, cv::COLOR_BGR2HSV);
    // split into planes
    std::vector<cv::Mat> planes;
    cv::split(img, planes);

    // only work on value-channel
    planes[2] = usm(planes[2], filterMode, size, thresh, scale);

    // merge planes to color image
    cv::merge(planes, img);
    // convert HSV to BGR
    cvtColor(img, img, cv::COLOR_HSV2BGR);
    // convert 32F to U8
    img.convertTo(img, CV_8UC3);

    return img;
}


float benchmarkMethod(const cv::Mat_<float> &image, dip3::FilterMode filterMode, int size)
{
    // warmup cache and branch predictor
    dip3::smoothImage(image, size, filterMode);

    // measure for real
    StopWatch stopWatch;
    const unsigned numIters = 1;
    for (unsigned i = 0; i < numIters; i++)
        dip3::smoothImage(image, size, filterMode);
    return stopWatch.getElapsedSeconds() / numIters;
}

using namespace std;
using namespace cv;

// usage: path to image in argv[1]
// main function. loads image, calls test and processing routines, records processing times
int main(int argc, char** argv) {

    // check if enough arguments are defined
    if (argc < 2){
        cout << "Usage:\n\tdip3 path_to_original"  << endl;
        cout << "Press enter to exit"  << endl;
        cin.get();
        return -1;
    }

    // load image, path in argv[1]
    cout << "Load image: start" << endl;
    Mat imgIn = imread(argv[1], IMREAD_COLOR);
    if (!imgIn.data) {
        cout << "ERROR: original image not specified"  << endl;
        cout << "Press enter to exit"  << endl;
        cin.get();
        return 0;
    }
    cout << "Load image: done" << endl;

    std::cout << "Showcasing functionality" << std::endl;

/*
    // some windows for displaying images
    const char* win_1 = "Degraded Image";
    const char* win_2 = "Enhanced Image";
    const char* win_3 = "Differences";
    namedWindow( win_1, CV_WINDOW_AUTOSIZE );
    namedWindow( win_2, CV_WINDOW_AUTOSIZE );
    namedWindow( win_3, CV_WINDOW_AUTOSIZE );
*/
    // distort image with gaussian blur
    int size = 5;
    GaussianBlur(imgIn, imgIn, Size(floor(size/2)*2+1,floor(size/2)*2+1), size/5., size/5.);
    cv::imwrite("dedgraded.png", imgIn);

    // show degraded
    cv::imshow("Degraded Image", imgIn);

    for (unsigned i = 0; i < dip3::NUM_FILTER_MODES; i++) {
        cv::Mat output = processColorImage(imgIn, (dip3::FilterMode) i, 5, 1.0f, 5.0f);
        
        cv::imwrite(std::string(dip3::filterModeNames[i])+".png", output);

        // show filtered
        cv::imshow(dip3::filterModeNames[i], output);
    }
    cv::waitKey(0);


    std::cout << "Running Benchmark" << std::endl;

    std::vector<unsigned> benchmarkImageSizes = {
        8,
        16,
        32,
        64,
        128,
        256,
        512,
        1024,
//        2048
    };

    std::vector<unsigned> benchmarkKernelSizes = {
        3,
        5,
        7,
        9,
        11,
        21,
        31,
        41,
        51,
        71,
        101,
/*
        201,
        401,
        801,
        1601
*/
    };

    for (unsigned i = 0; i < dip3::NUM_FILTER_MODES; i++) {
        std::string filename;
        filename = "benchmark_";
        filename += dip3::filterModeNames[i];
        filename += ".csv";

        std::cout << "Writing results for " << dip3::filterModeNames[i] << " to " << filename << std::endl;
        std::fstream csvFile(filename.c_str(), std::fstream::out);
        
        csvFile << "Execution time in seconds for " << dip3::filterModeNames[i] << ";Image sizes in rows;Kernel sizes in columns" << std::endl;
        for (unsigned i : benchmarkKernelSizes)
            csvFile << ";" << i;
        csvFile << std::endl;

        for (unsigned imgSize : benchmarkImageSizes) {
            cv::Mat_<float> image = cv::Mat_<float>::zeros(imgSize, imgSize);
            csvFile << imgSize;
            for (unsigned kernelSize : benchmarkKernelSizes) {
                if (kernelSize > imgSize) {
                    csvFile << ';';
                    continue;
                }
                std::cout << "Benchmarking " << dip3::filterModeNames[i] << " on " << imgSize << "^2 pixel image with " << kernelSize << "^2 pixel kernel" << std::endl;
                csvFile << ';' << benchmarkMethod(image, (dip3::FilterMode)i, kernelSize);
            }
            csvFile << std::endl;
        }

    }
  
    

   return 0;
} 
