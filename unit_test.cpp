//============================================================================
// Name        : main.cpp
// Author      : Ronny Haensch
// Version     : 2.0
// Copyright   : -
// Description : only calls processing and test routines
//============================================================================


#include "Dip2.h"

#include <opencv2/opencv.hpp>

#include <iostream>

#include <random>

using namespace std;
using namespace cv;
using namespace dip2;


// checks basic properties of the convolution result
void test_spatialConvolution(void){

   Mat input = Mat::ones(9,9, CV_32FC1);
   input.at<float>(4,4) = 255;
   Mat kernel = Mat(3,3, CV_32FC1, 1./9.);

   Mat output = spatialConvolution(input, kernel);
   
   if ( (input.cols != output.cols) || (input.rows != output.rows) ){
      cout << "ERROR: Dip2::spatialConvolution(): input.size != output.size --> Wrong border handling?" << endl;
      exit(-1);
   }
  if ( (sum(output.row(0) < 0).val[0] > 0) ||
           (sum(output.row(0) > 255).val[0] > 0) ||
           (sum(output.row(8) < 0).val[0] > 0) ||
           (sum(output.row(8) > 255).val[0] > 0) ||
           (sum(output.col(0) < 0).val[0] > 0) ||
           (sum(output.col(0) > 255).val[0] > 0) ||
           (sum(output.col(8) < 0).val[0] > 0) ||
           (sum(output.col(8) > 255).val[0] > 0) ){
         cout << "ERROR: Dip2::spatialConvolution(): Border of convolution result contains too large/small values --> Wrong border handling?" << endl;
          exit(-1);
   }else{
      if ( (sum(output < 0).val[0] > 0) ||
         (sum(output > 255).val[0] > 0) ){
            cout << "ERROR: Dip2::spatialConvolution(): Convolution result contains too large/small values!" << endl;
          exit(-1);
      }
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
            cout << "ERROR: Dip2::spatialConvolution(): Convolution result contains wrong values!" << endl;
           exit(-1);
         }
      }
   }
   input.setTo(0);
   input.at<float>(4,4) = 255;
   kernel.setTo(0);
   kernel.at<float>(0,0) = -1;
   output = spatialConvolution(input, kernel);
   if ( abs(output.at<float>(5,5) + 255.) < 0.0001 ){
      cout << "ERROR: Dip2::spatialConvolution(): Is filter kernel \"flipped\" during convolution? (Check lecture/exercise slides)" << endl;
     exit(-1);
   }
   if ( ( abs(output.at<float>(2,2) + 255.) < 0.0001 ) || ( abs(output.at<float>(4,4) + 255.) < 0.0001 ) ){
      cout << "ERROR: Dip2::spatialConvolution(): Is anchor point of convolution the centre of the filter kernel? (Check lecture/exercise slides)" << endl;
      exit(-1);
   }
   cout << "Message: Dip2::spatialConvolution() seems to be correct" << endl;
}

// checks basic properties of the filtering result
void test_averageFilter(void){

   Mat input = Mat::ones(9,9, CV_32FC1);
   input.at<float>(4,4) = 255;

   Mat output = averageFilter(input, 3);
   
   if ( (input.cols != output.cols) || (input.rows != output.rows) ){
      cout << "ERROR: Dip2::averageFilter(): input.size != output.size --> Wrong border handling?" << endl;
      exit(-1);
   }
  if ( (sum(output.row(0) < 0).val[0] > 0) ||
           (sum(output.row(0) > 255).val[0] > 0) ||
           (sum(output.row(8) < 0).val[0] > 0) ||
           (sum(output.row(8) > 255).val[0] > 0) ||
           (sum(output.col(0) < 0).val[0] > 0) ||
           (sum(output.col(0) > 255).val[0] > 0) ||
           (sum(output.col(8) < 0).val[0] > 0) ||
           (sum(output.col(8) > 255).val[0] > 0) ){
         cout << "ERROR: Dip2::averageFilter(): Border of result contains too large/small values --> Wrong border handling?" << endl;
         exit(-1);
   }else{
      if ( (sum(output < 0).val[0] > 0) ||
         (sum(output > 255).val[0] > 0) ){
            cout << "ERROR: Dip2::averageFilter(): Result contains too large/small values!" << endl;
            exit(-1);
      }
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
            cout << "ERROR: Dip2::averageFilter(): Result contains wrong values!" << endl;
            exit(-1);
         }
      }
   }
   cout << "Message: Dip2::averageFilter() seems to be correct" << endl;
}

// checks basic properties of the filtering result
void test_medianFilter(void){

   Mat input = Mat::ones(9,9, CV_32FC1);
   input.at<float>(4,4) = 255;

   Mat output = medianFilter(input, 3);
   
   if ( (input.cols != output.cols) || (input.rows != output.rows) ){
      cout << "ERROR: Dip2::medianFilter(): input.size != output.size --> Wrong border handling?" << endl;
      exit(-1);
   }
  if ( (sum(output.row(0) < 0).val[0] > 0) ||
           (sum(output.row(0) > 255).val[0] > 0) ||
           (sum(output.row(8) < 0).val[0] > 0) ||
           (sum(output.row(8) > 255).val[0] > 0) ||
           (sum(output.col(0) < 0).val[0] > 0) ||
           (sum(output.col(0) > 255).val[0] > 0) ||
           (sum(output.col(8) < 0).val[0] > 0) ||
           (sum(output.col(8) > 255).val[0] > 0) ){
         cout << "ERROR: Dip2::medianFilter(): Border of result contains too large/small values --> Wrong border handling?" << endl;
         exit(-1);
   }else{
      if ( (sum(output < 0).val[0] > 0) ||
         (sum(output > 255).val[0] > 0) ){
            cout << "ERROR: Dip2::medianFilter(): Result contains too large/small values!" << endl;
            exit(-1);
      }
   }
   for(int y=1; y<8; y++){
      for(int x=1; x<8; x++){
         if (abs(output.at<float>(y,x) - 1.) > 0.0001){
            cout << "ERROR: Dip2::medianFilter(): Result contains wrong values!" << endl;
            exit(-1);
         }
      }
   }
   cout << "Message: Dip2::medianFilter() seems to be correct" << endl;

}

extern const std::uint64_t data_inputImage[];
extern const std::size_t data_inputImage_size;

namespace {

float computePSNR(const cv::Mat_<float> &estimate, const cv::Mat_<float> &orig)
{
    cv::Mat_<float> diff = estimate - orig;
    float meanSqrDiff = cv::mean(diff.mul(diff))[0];
    float PSNR = 10.0f * std::log10(255*255 / meanSqrDiff);
    return PSNR;
}

cv::Mat_<float> generateNoisyImage(const cv::Mat_<float> &img, dip2::NoiseType noiseType)
{
    // generate images with different types of noise
    switch (noiseType) {
        case dip2::NOISE_TYPE_1: {
            // some temporary images
            cv::Mat tmp1(img.rows, img.cols, CV_32FC1);
            cv::Mat tmp2(img.rows, img.cols, CV_32FC1);
            // first noise operation
            float noiseLevel = 0.15;
            cv::randu(tmp1, 0, 1);
            cv::threshold(tmp1, tmp2, noiseLevel, 1, THRESH_BINARY);
            cv::multiply(tmp2,img,tmp2);
            cv::threshold(tmp1, tmp1, 1-noiseLevel, 1, THRESH_BINARY);
            tmp1 *= 255;
            tmp1 = tmp2 + tmp1;
            cv::threshold(tmp1, tmp1, 255, 255, THRESH_TRUNC);
            return tmp1;
        } break;
        case dip2::NOISE_TYPE_2: {
            // some temporary images
            cv::Mat tmp1(img.rows, img.cols, CV_32FC1);
            cv::Mat tmp2(img.rows, img.cols, CV_32FC1);
            // second noise operation
            float noiseLevel = 50;
            cv::randn(tmp1, 0, noiseLevel);
            tmp1 = img + tmp1;
            cv::threshold(tmp1,tmp1,255,255,THRESH_TRUNC);
            cv::threshold(tmp1,tmp1,0,0,THRESH_TOZERO);
            return tmp1;
        } break;
        default:
            throw std::runtime_error("Unhandled noise type!");
    }
}

}


// checks basic properties of the filtering result
void test_bilateralFilter()
{
    {
        cv::Mat_<float> input = cv::Mat_<float>::ones(15, 15);
        cv::Mat_<float> output = bilateralFilter(input, 5, 0.2f, 1.0f);
        if ( (input.cols != output.cols) || (input.rows != output.rows) ){
            cout << "ERROR: Dip2::bilateralFilter(): input.size != output.size --> Wrong border handling?" << endl;
            exit(-1);
        }
    }

    {
        cv::Mat_<float> input = cv::Mat_<float>::ones(15, 15);
        cv::Mat_<float> output = bilateralFilter(input, 5, 0.2f, 1.0f);

        for (unsigned y = 5; y < 10; y++)
            for (unsigned x = 5; x < 10; x++) {
                if (std::abs(output(y, x) - 1.0f) > 1e-3f) {
                    cout << "ERROR: Dip2::bilateralFilter(): completely homogeneous image gets changed. Wrong normalization?" << endl;
                    cout << "    Remember you need to divide by the sum of all weights (each weight being the product of the spatial and radiometric factors)." << endl;
                    exit(-1);
                }
            }

    }

    {
        std::mt19937 rng;
        std::normal_distribution<float> dist(127.0f, 1.0f);

        cv::Mat_<float> input(130, 130);
        for (unsigned y = 0; y < input.rows; y++)
            for (unsigned x = 0; x < input.cols; x++)
                input(y, x) = dist(rng);

        cv::Mat_<float> output = bilateralFilter(input, 51, 10.0f, 10000.0f);
        for (unsigned y = 60; y < 70; y++)
            for (unsigned x = 60; x < 70; x++) {
                if (std::abs(output(y, x) - 127.0f) > 1e-1f) {
                    cout << "ERROR: Dip2::bilateralFilter(): Filtering with very large radiomatric sigma (so only the gaussian filter part) not effective." << std::endl
                         << "     Some error in the spatial factors or the summation?" << endl;
                    exit(-1);
                }
            }
    }

    {
        cv::Mat_<float> input(130, 130);
        for (unsigned y = 0; y < input.rows; y++)
            for (unsigned x = 0; x < input.cols; x++)
                input(y, x) = (x > 130/2?0.0f:255.0f);

        cv::Mat_<float> output = bilateralFilter(input, 51, 10.0f, 0.001f);

        for (unsigned y = 60; y < 70; y++)
            for (unsigned x = 60; x < 70; x++) {
                if (std::abs(output(y, x) - input(y, x)) > 1e-3f) {
                    cout << "ERROR: Dip2::bilateralFilter(): Radiometric factors don't prevent blured edges." << std::endl
                         << "     Some error in the radiometric factors?" << endl;
                    exit(-1);
                }
            }
    }
   cout << "Message: Dip2::bilateralFilter() seems to be correct" << endl;

}


void test_denoiseImage()
{
    cv::Mat img = cv::imdecode(cv::_InputArray((const char *)data_inputImage, data_inputImage_size), 0);
    img.convertTo(img, CV_32FC1);

    cv::Mat noise[2] = {
        generateNoisyImage(img, dip2::NOISE_TYPE_1),
        generateNoisyImage(img, dip2::NOISE_TYPE_2)
    };

    float expectedPSNRs[dip2::NUM_NOISE_TYPES][dip2::NUM_FILTERS] = {
        {17.5f, 21.0f, 17.5f},
        {21.0f, 20.0f, 22.0f},
    };

    for (unsigned i = 0; i < dip2::NUM_NOISE_TYPES; i++)
        for (unsigned j = 0; j < dip2::NUM_FILTERS; j++) {
            float psnr = computePSNR(dip2::denoiseImage(noise[i], (dip2::NoiseType) i, (dip2::NoiseReductionAlgorithm) j), img);
            if (psnr < expectedPSNRs[i][j]) {
                std::cout << "ERROR: Dip2::denoiseImage(): Expected better PSNR on internal test image for noise " 
                          << noiseTypeNames[i] << " with algorithm " << noiseReductionAlgorithmNames[j] << std::endl;
                std::cout << "     achieved " << psnr << "dB, but expected at least " << expectedPSNRs[i][j] << "dB" << std::endl;
                exit(-1);
            }
        }

   cout << "Message: Dip2::denoiseImage() seems to be correct" << endl;

}


int main(int argc, char** argv) {
    test_spatialConvolution();
    test_averageFilter();
    test_medianFilter();
    test_bilateralFilter();
    test_denoiseImage();

	return 0;
} 







const std::uint64_t data_inputImage[] = {
   0xa1a0a0d474e5089ul, 0x524448490d000000ul, 0x78000000a0000000ul, 0xa6c9500000000008ul, 0x43436923010000b3ul, 0x6f72702043434950ul, 0x91280000656c6966ul, 0x861450c34ab1909dul, 0x838a20ea52a954bful, 0xa5c998b82d706608ul, 0xa7560ac62141042aul, 0x14a490c498b14934ul, 0x820e987d137c0dful, 0xa37fece0a06f80beul, 0xe3ff870bc598383ul, 0x12765a17bdfffce7ul, 
   0xb85559a42ee2e5a6ul, 0xde97b2bc39787f7eul, 0x5841ba6d2acdb0e8ul, 0xbe7cf1a13bcf7de6ul, 0xb9e6af19e97d1962ul, 0x573a50cb8a3b4f3ful, 0x9d8bed60545e6165ul, 0xf03b7eb1561b9559ul, 0x48b34a3b620fc50ful, 0xd9b0c8d289de24fcul, 0xdb9a78fe1a64d3f5ul, 0x5b55f4dce2ece374ul, 0x2988cd878a731cb8ul, 0x239d4cd27a2a1213ul, 0x94f701052ea4f61cul, 0x2a6699bd5884d284ul, 
   0x3440e5c9ca5446eul, 0x79e759b790d36e91ul, 0x70932f2263c9194aul, 0xefdff987934f2a47ul, 0xe798dad37ab38fb5ul, 0x3c6b5505add41141ul, 0x33dac2195847f786ul, 0xadbf7f96b21bae74ul, 0x2fc6f9fe67a9c661ul, 0x6eaefd1d4f5034bdul, 0x7359487009000000ul, 0xd4350000d4350000ul, 0x20000008e5655e01ul, 0x6dda785441444900ul, 0x39df75e72593d77cul, 0x7771c9df39bee75ful, 
   0x112258bbb16136ul, 0xb2b2ca28a4880601ul, 0x5f965c96cb83f254ul, 0x55d1f952e03ff65dul, 0x3255483f2595707eul, 0x48942a44ba2adb2dul, 0xd8b177632200948aul, 0x1cdcee6793b3b3bcul, 0xcf5f0edc3f1f773aul, 0xa7dbdc277bd83062ul, 0xcbe0df39df9dfc4ful, 0xe9b3e10088008000ul, 0xfe1004004040817ul, 0xc08800008fa7110ul, 0xe980021dfc621910ul, 0x87fa291e4a0fc26bul, 
   0xa2f0c1dfa77fa28ul, 0xa11397e2880065f1ul, 0x7f4d0210021dbdc8ul, 0xdf8c2dfc52bb909ful, 0x404cc918f859010dul, 0x2122b2a74bf148ul, 0x1062867d144a2f4dul, 0xebd3748912e89977ul, 0x77d3789104309c89ul, 0x457bf8c40080b58dul, 0x4fe1848a950c440ul, 0x537d4499aa700909ul, 0x43315394890f4595ul, 0x41f4df2f24bc3121ul, 0x424d431aee8bc50cul, 0x483722c47558a578ul, 
   0x91a6ca67e1088654ul, 0x12cfc5f2f3d7927dul, 0xb7f8906a163744e3ul, 0x226e8c2477792112ul, 0x145380099a9c40f4ul, 0x4e07dc07148c0ac6ul, 0x899ce40301008d6ful, 0x7a9c13725130b7ul, 0xa8620878de040210ul, 0x7123356339d8c032ul, 0x2054381a17091231ul, 0x705c7c20e8702861ul, 0x17b167a20c22247aul, 0x3024a1375341af4eul, 0x88c6a18810224d63ul, 0xc24f452fa9a46762ul, 
   0x522caad082c48da0ul, 0xc648c55745362c62ul, 0x12301d88c0f855e9ul, 0x4004076e72273225ul, 0x325b1c4d0d80d384ul, 0x44eb8a30fa288f0aul, 0xbcde43bac5d7a15dul, 0x1057d14ff54dcbb1ul, 0x20037b894c61f120ul, 0x7753ff5e2ff53061ul, 0x524c3cf628295314ul, 0x84cf30c039a8610bul, 0xb123e67fd652c1c1ul, 0x85b44e1653069c6ful, 0x422b7727e049ffd1ul, 0xc12988457786113aul, 
   0x310c14302894aca3ul, 0x184e071a6f9828ceul, 0xbbc074728e0f5365ul, 0xe4616312015711ful, 0x38dccf0b0bcd2a76ul, 0xccd3880289c0e1f2ul, 0x7ae50a0cf5321532ul, 0xc472a67434139067ul, 0x866fe24555209489ul, 0x558a45646b33e62aul, 0x58be171880acf489ul, 0xd8987fec4b338a2eul, 0x7721656c941001dul, 0x4be3f1d556fa1043ul, 0x2e8525f524e36713ul, 0xe58be26af63002faul, 
   0xa795545b53906a17ul, 0x1b1222a23d5c594ful, 0x3b27e8feef77fcfbul, 0x79ef3f32cf40e695ul, 0xf88ff92de570dbe1ul, 0xf440e04c5da7131aul, 0xa329d202538c5618ul, 0x65ba2b912c06cef2ul, 0x3adeffffbf280040ul, 0xbda68e2a164a5bful, 0x7b915caf2a9bae48ul, 0xf888f58512243c06ul, 0x8821530c0000dacul, 0xf7c5a6b8a5c66180ul, 0xc6e100a4c3e71044ul, 0xc807993c14bf8777ul, 
   0x62921dc1dce426a3ul, 0x8082888f1d4d11dul, 0xa994ef1207388288ul, 0x3a78354f2c5c8952ul, 0xc4889151af146049ul, 0xedf16ee4987958c2ul, 0x8e73cb26379bfb0ful, 0xd13f99f5810b5423ul, 0x6b627372e27f8f88ul, 0xd7175c932de89e4eul, 0x77c4d93a59928806ul, 0x7c6125e72010ba95ul, 0x67617cff0ff66010ul, 0x68144175bfd03ce8ul, 0x21c18d9e514263e7ul, 0x34e22014154d8625ul, 
   0xc854c28fc8b867c1ul, 0xc2f611b24f8428d2ul, 0xd9442108a954b644ul, 0xfaef02b1aa2a9673ul, 0x59155f7d07dbfd85ul, 0x2c42a718d7c3cb35ul, 0x9aa7aef887af2531ul, 0x10d9ec78d4df2b8cul, 0xa632d24c5c7ecb0eul, 0x8caf4beaee784995ul, 0x48c29bdc14d9c915ul, 0xf022e1e9c32d15acul, 0x12a277b4325c4bbaul, 0x45e8b4574a186716ul, 0x8d7f8d156181e52cul, 0xc28250cd0a76b22ul, 
   0x16137df7a303f41ul, 0xd671a5d03f16ee7eul, 0x2518a646d6f096cul, 0x7e117429f15e404ul, 0xf3a71d58b6a017e1ul, 0x7803e5c6026dc620ul, 0xc7fb15fe337e131cul, 0x404abd4bb25f6627ul, 0xc8dd5337b8cd78f2ul, 0x257c8c1530ee9465ul, 0xea1c88061c7d124dul, 0x494899a7486414c0ul, 0x45774a6a62763bc8ul, 0xc66efe3fbbd40000ul, 0x94da500c9958fb13ul, 0xe04733e0bad55baaul, 
   0xc49411825e2616aaul, 0x6ced3c25a6514944ul, 0x1594507054e8c000ul, 0xeda8c61a310f4ae2ul, 0xd4a0bafa2757220aul, 0x3b6dc656d0e4ba2aul, 0x23fb5b93f90cf77bul, 0x182e0345a436d55ul, 0x4002cb4e0dfd0bf0ul, 0x900f6d3dbfa52228ul, 0x65a01142812b2eb8ul, 0x3950b4b0154c1084ul, 0xbbd95ef6120883c9ul, 0x7bd3fcb5de7a10dbul, 0xe9cb793744963ee7ul, 0x2d00b3e7c71060bbul, 
   0xc25bc12a0ed3c742ul, 0x2a291ad94c636f22ul, 0x404241c048f7181bul, 0x6528d4f4b85e7a00ul, 0x47e5ea7e51f7c0bcul, 0x3457153c16dae5c2ul, 0x21bf4a9c1136e533ul, 0x374d381e33f73d26ul, 0x8e5ce420c31334a9ul, 0x1158418bf476ed52ul, 0x6d67be3bed9da58aul, 0x583e58ecfada9259ul, 0xa8a0865b590be958ul, 0xa8cf56374f1c8700ul, 0xb4e0c0a1712db784ul, 0x42e72e62e117d007ul, 
   0x7d6e8fb9e10378d2ul, 0x4c04cf283d0346ful, 0x65376c1a0760317cul, 0xd53ff7c30c8405a6ul, 0xe44c9778289f116eul, 0x21d0e0458513238ful, 0x314c46667070cc50ul, 0xd9fd1f010856b11ful, 0xa84b53d2fd2039bful, 0x3d4375d6cf424b2bul, 0x77222db96cc27131ul, 0x1aa99ac6338053c5ul, 0xb304b33018cd249aul, 0x5c5c60a9cd263544ul, 0x40a5f8fc293c0e92ul, 0xaf881d92b0fb3492ul, 
   0x32db8670446616e5ul, 0xc888c7b4e8cf44b3ul, 0x9ee357e93011e930ul, 0x5a914c4a9081c398ul, 0xfba8c362639a04c4ul, 0x4261034d4622cb8cul, 0x6b517c5f152109bul, 0x99ab7970f1464736ul, 0x91c5bb04b77ded53ul, 0xa24261c93ec8d2e4ul, 0x440f4842d30cba44ul, 0xb1d74a74fe813f15ul, 0x18d6cb88fbea3189ul, 0x1099611471185ac7ul, 0xfeaf03f61716ad6dul, 0x4796b49a5ed76499ul, 
   0xba360491ac67a042ul, 0xb3a4e43aa5cf1f60ul, 0xde29c140453fdfe1ul, 0x84c01098c584110aul, 0x9292df4f5a18c12cul, 0x51be22023afb1084ul, 0xa811f7ad78e1ffbful, 0xe2bde1d2b92ca48eul, 0xf3c2f647d78bf79eul, 0x390fc053f7988acaul, 0x33d424c5bb3a41d3ul, 0x5945db0e02584ec9ul, 0x90362a618f11208aul, 0x7df5ff8b8de090e3ul, 0x1bfb9aecb607b532ul, 0xd5ff365d27e9fc18ul, 
   0x3d4233c8ab516b79ul, 0x8242fb0e10398bdful, 0x19f098f0951309caul, 0x631ac8722cd1a713ul, 0x56559a510931ca80ul, 0xb8237607c15df2feul, 0x7cb71ffb1ded4708ul, 0x2b6696b20a5e2061ul, 0xa46d426ec6f39096ul, 0xb22a605d24841388ul, 0x9945fa4f11820c93ul, 0x3479109953e36f9aul, 0xfc8eabcb15731671ul, 0x9247fadbd37c9dc5ul, 0x2ac86db9c7938400ul, 0x6013e711a54a89a4ul, 
   0x7714d8f47190c808ul, 0x8c45080daa7f2e26ul, 0xc5a789849ced2483ul, 0x52e1787f4e3c93deul, 0x9f2a385d91e687edul, 0x366b0eeb92270613ul, 0x3e2b1655b97b7cceul, 0xfe28a43fc56e040ul, 0x22cbc26252393ec2ul, 0xb1be311ba1213807ul, 0x91934c061a48f7a4ul, 0xf6ab7d7a9e4e1714ul, 0x283cd68ee1252700ul, 0x650d87781caefdeaul, 0x87c3ed2b71368ceeul, 0x71925a9f92801041ul, 
   0x54fa9da625a12b00ul, 0x390dc64795d53a68ul, 0x9875c5c49ab14c6aul, 0x57d647a23836f8f5ul, 0x959c7ed55183f27ful, 0xb4466307e5a0fcacul, 0x2be5ad4ff9ffdf37ul, 0x9daa0f96b7e11a3ful, 0xf17d3a4314e0d527ul, 0x639417eaf34d2f14ul, 0xd617e462d7c0388eul, 0xe878d10844c90d8ul, 0xff55959e62f942c3ul, 0xead8d9d79dfeffe0ul, 0x9fd83c753414aeul, 0xcb2555f1dca09300ul, 
   0x452e8014edaf5a72ul, 0x153f499fb710675dul, 0x6145a8f83366957ful, 0xab8a01f5344a1d29ul, 0xe5f83ca9d9875c7aul, 0x5cbefd60386dff5bul, 0x80c1d72f750e2cc9ul, 0x2f1a21333d5cece1ul, 0x92836ae7aebf13d2ul, 0x1c2ae380090ac690ul, 0x808880caa9dec446ul, 0x2be7da8a4efc73aful, 0xac0406281c603028ul, 0x5ebe41c8009e3f74ul, 0xa9b53ed2dd06f25dul, 0xf23c74ee6fba0b3eul, 
   0x9135f45e6852631aul, 0x34e024104446b062ul, 0x140b0b141c934569ul, 0x8408b765c4b0a4c6ul, 0x14e98ee14527edc6ul, 0x290408d6f9d98089ul, 0x57f6232168e19573ul, 0xa286ad258fcc6d49ul, 0x3d5f42440a8110d7ul, 0x8a243713ff38bccbul, 0x33179d3c9f723571ul, 0x9452110e5f32c255ul, 0xb2a3dd2489729e3bul, 0x115cd26b6781fe70ul, 0x8717060979a1d0abul, 0xb342f41e632cbd66ul, 
   0xb160aa19269aa824ul, 0x3a438801154ab014ul, 0x2c2f8145db04ef0eul, 0x4b746fd00bec68c5ul, 0x6f7d7eb86aa771bcul, 0x56ec5c4d475559d3ul, 0x56cfc16a955b0fd7ul, 0x2ae5a40987230a27ul, 0x86a5b6a80adafe76ul, 0x453fdc90ace29070ul, 0x620110c03bf12ab5ul, 0xa491e3c7f2cc6428ul, 0x264eb294d3cdc476ul, 0x70ed7edccfa0081aul, 0x9ed7abdd9b9a91ebul, 0xaee69daa3190be6ul, 
   0xf773816540702e66ul, 0x13b20749cbd00dc4ul, 0x700bc3e36e9061f5ul, 0x18bab42dc4d4a0baul, 0xc2425fc34e90a04ful, 0x826921488a9fe598ul, 0xece775dcebf83f97ul, 0xb196244fb0dc5d5cul, 0x17ee2b3c64cc9d0ful, 0x787cf777bb103f24ul, 0x824e8795295fc5c7ul, 0x904496f09d0023d7ul, 0xc556a99595c01baaul, 0x53861aa4a24607b0ul, 0x448a69b9ad394ul, 0x17e4af5793f8c378ul, 
   0x528985a48b7db8ful, 0x9dd92a17df92ab7ful, 0x80017b1824062db2ul, 0xde1dbe72fe1f392dul, 0x8a1898a2403d387dul, 0xc911a11c48331b55ul, 0x4bbb929a9571122dul, 0xb0f6858e0e030a8aul, 0x4acbefeffd0ff9deul, 0x646bb73881fd6afeul, 0xbd02376a055980d7ul, 0xced9218a6156fa2ful, 0xc476100b2d5ee953ul, 0x6a54bdd384295518ul, 0x623d7c424de090beul, 0x54cfac250ba9e241ul, 
   0x37e7f22c5c062ccful, 0xf5a2498c61bd3f97ul, 0x9e667f95e0bb9bd7ul, 0x5e8ea2f98eeda96dul, 0x707f6ac64e29889ful, 0x98a02d72e8ed7982ul, 0x8949524429a20f44ul, 0xcd1087a2e113466ful, 0x55084ebe3f310c70ul, 0xdffb3efe2c45768eul, 0x9afabe52ba974bbbul, 0xce3136a6f26aadbbul, 0xd14c913528ef3798ul, 0x6bab8dead4dca153ul, 0x53274262c753a0adul, 0x12c462092ba56bc2ul, 
   0x747907d4d524720ul, 0xe217e8de8e3728fcul, 0xb12bb9ffabe67eb2ul, 0x98dfaf8e0f95fe51ul, 0xe4342fd77b43bdd1ul, 0x64819f045664f50dul, 0x10dff9adf47cc2a0ul, 0x379d4fe4ffd9af77ul, 0x939e01f01b885121ul, 0x62b7305f8a91856ful, 0xcc9f266a74471f51ul, 0x73be110f6a388855ul, 0xf071e56d4cfd16dful, 0x9a3eeda1f373b65aul, 0x8c28ffd5fd4c8d20ul, 0xb6073677a3fd920dul, 
   0xeb935c2d93b8a24ful, 0xa67f8fc01cd8b22aul, 0xa63659b86737b0c2ul, 0x130b022314005278ul, 0x4e1fdf21ac878c0ful, 0x72bcf9517980a2c6ul, 0x350fb06ca3f5c824ul, 0x8af4114e6d2d9ddul, 0xef4a991bd951c2dbul, 0x8838101abacee367ul, 0xc5303108d94aca1eul, 0xa8fe4e884a24f923ul, 0x8486b13d1f1f7a14ul, 0x51c3f0690a7a64c5ul, 0xb0ffafe6c2215cfcul, 0xbe425bec4e54cde7ul, 
   0x250caf576bdcae27ul, 0x5fdb94778fce49e6ul, 0x54cfc9f59f3b0712ul, 0xab2aa64a66bed6fdul, 0xa1f1cce06a51c6ul, 0xdf23938c05c71350ul, 0x1134522033074c9ful, 0x11dacdc8e5e64a5ul, 0xfeefc408fee8c020ul, 0xcc86327331d23e0ul, 0xcc2ad8c78fabcff6ul, 0xb87fce66fbaf9ddeul, 0x6ff005dea99cdd2aul, 0xe088a6bcb0cec6f9ul, 0x9bf67897dee79baul, 0xf9f94c98ab72bf20ul, 
   0x24939a25b8862f01ul, 0xa1e9ca986558a1aful, 0x95e6bbb4ffabfcebul, 0xf1336c417d927f4ul, 0x401c0467cf6b8d9bul, 0xe8cf3fd62863bd03ul, 0xac1647877809c333ul, 0xfde59b87fcd1819cul, 0x7b2e713126221553ul, 0xce26a8241640482ful, 0x6e3f3e6d702c201aul, 0x2edf88c6e50e8e45ul, 0x6af15b5dce1b9f8cul, 0x2d92c7e3d5f879bul, 0xfbf377a7be0ae713ul, 0xa659b5daf8fa453dul, 
   0xc9a5dae0b658c7ebul, 0xeb519b1dc9feb77bul, 0xf335c40243b03c6ul, 0x4405a422741b880ul, 0x8919a39279d21120ul, 0xe16b7ecf62c255ebul, 0x95839373eb5b0be3ul, 0x76b93ec77c68fceeul, 0xf86e9140cb3ecbaful, 0x2a15c97b5d1e4579ul, 0xc2caf64560f73639ul, 0x122e6ab41e807ef1ul, 0xb1a12310012335cful, 0x1021714e86058767ul, 0x2826e63069185653ul, 0x7d911100ffc88cc6ul, 
   0xfb7cc9dc3fdbd7f3ul, 0x74a00e4e963f77cul, 0x1e83175d0bd51ec3ul, 0x757954756f32efedul, 0x508ab31b24a09f51ul, 0xb8767abc283dc081ul, 0x21ae99309f6a9938ul, 0xa78b5f5cfb694640ul, 0x2d48450f96c20d56ul, 0xe734e872e231211aul, 0x9fa191fd1cda3e44ul, 0x730f9bcae77ebbd8ul, 0xabbd4a730bcd07b7ul, 0x9d518cddcb97e3a0ul, 0x9a97df927b55b45cul, 0xfcd5d2c406c641deul, 
   0xa690908c12ab42caul, 0x8ee72c4a0b85265cul, 0xb7f5c4c6c2938039ul, 0xc1cad754e8106bc4ul, 0xe0ae7effb5584072ul, 0xa8f27de5828a6656ul, 0x8347d52bf41caef4ul, 0x3518d159f886fa99ul, 0x86f7facb5046b7aaul, 0x84ba7b6f66b7878dul, 0x18339fb47146466aul, 0x4e2723ca506d99a4ul, 0xbaa2bf639a08c9cbul, 0x9072b40e5093f225ul, 0xa3cca59a96efea26ul, 0x2270f8182818aeebul, 
   0x7f52f5c68bc3e5a9ul, 0xee72fae1f08ef0beul, 0x333a8147b19bbcf3ul, 0xc50055b5d363ef74ul, 0x9090733240d93af0ul, 0xd52088e9150848a3ul, 0x529a222b6a9e1060ul, 0xe13064925fcae458ul, 0xaad982faf371e292ul, 0xf9b9977378c68caeul, 0x7cfedb59df4eb176ul, 0xe3c6d42fbfbab8b0ul, 0xfbbb3968667927d9ul, 0xfe7e3fdd7c1ddfdeul, 0xaaa39ad3e4c28f28ul, 0xb5083251c0b3f7cbul, 
   0xf838adcfcfb47f7ful, 0x53c4c098b7e8a780ul, 0xaae111071553b463ul, 0x206ee48fba57f50bul, 0x7fb19cdcb9499d79ul, 0x7d54aea9b7832bdcul, 0xcc3958b856bfdc79ul, 0xe0f571e3389e1c3ful, 0xe6e7bbe3e023b352ul, 0x3896b88676e6c39dul, 0x1550e0376b73bd9ful, 0x45b5c949ab348030ul, 0x78c202df8ce7667cul, 0x97381faff5690d29ul, 0xab2bbffb4ab3a5c6ul, 0x9b7c57172e19dd26ul, 
   0xd4cf3559be95e6d4ul, 0x9ba662689fd6db83ul, 0x50e2ce5c2f0ca96ful, 0x21a72661045b5f54ul, 0x8339fdd86c59b8a3ul, 0x5525309d4505fc45ul, 0xf0e41c280150662cul, 0x8f85789a669c0cf3ul, 0x39db00ffcfcc3342ul, 0x2b5c2ccf70f47b76ul, 0x4b33f78d2c6b42fdul, 0xf596a4aa57ac2db3ul, 0x8fcedc496f59ffeful, 0x2825f9f2b34b2766ul, 0xb070691416574ab4ul, 0xdc6183aa0976e6d8ul, 
   0xf4e85b9c706d25a1ul, 0xea78f86944b09aceul, 0x54bb13f488b9c231ul, 0x5d79e60baead9daful, 0x740cc567c96a3c28ul, 0xdf2c7be0b3759b8ful, 0x15eb71f5ec8c29dful, 0x50783c586c6b2d4bul, 0x589d1dcf2cfff684ul, 0x498d67832842381ul, 0x973093aab263e41eul, 0xa387c972204d0b09ul, 0xfb9c1fd3234b53daul, 0x68f5131a4f320fcaul, 0x264a22b19d467522ul, 0x2ca977237aa65a19ul, 
   0x45717fbfb3e30394ul, 0x906619cb9e36bcdful, 0xbc7c6fb41feb52cdul, 0x8df7b5ecef6c99ful, 0x8644f18ac93d94e1ul, 0x59aa4ef0a20de0c2ul, 0x1e9409ec62474cb2ul, 0xb9ba0211c74f10a2ul, 0xe77113c68ba5fbb7ul, 0x516acacdcb573961ul, 0xcb833693d77df5ecul, 0x5e2989ccd3fedf98ul, 0x9d966dddca66cd9cul, 0xd398d57a7e1f6e81ul, 0x12c2d83884e5becdul, 0x2db80aec3c771d7ul, 
   0xb7642ff071eb2e98ul, 0x533d2b24104cc530ul, 0xc9783bce6f864637ul, 0x1ed87d61f77969faul, 0xe6672791fdb9ef1aul, 0xe1a1a3034a163667ul, 0x9c16776a5abe56b6ul, 0x9babdf461625a149ul, 0x497bb85d4ced7f65ul, 0x22be0413be28ecdaul, 0xd4a45843d3a466c5ul, 0xa711961261644bb6ul, 0x1938b26ea9a98511ul, 0x1b27f74ff6f010b2ul, 0xeab91d6f563aff92ul, 0xbbe7e6fee9adb6a4ul, 
   0x8fce4e42e1cde2adul, 0x87ff5da007cb6773ul, 0x6333f143abfb8d87ul, 0x1c9f644d736aa2cdul, 0xb9809378e07a3c6cul, 0x70f0613a41f8a6baul, 0xbd312f3384bd4d2eul, 0xd22840c99fb11007ul, 0x9dddb41eefbb22ecul, 0x319ba95e4dbebb3bul, 0x9ea0a99d89d5e7b3ul, 0x5b273db95bcb1e19ul, 0xe0ffb247e5ee6792ul, 0x4366717a6ba99eccul, 0x9460db32179346b6ul, 0xfc4fa921ab034a16ul, 
   0x1d9e80b05f1be2d4ul, 0xe2c6eff179ebf368ul, 0x77becc0022053c45ul, 0xeb27e82e86c3fb28ul, 0xb0dabfc67b3ba627ul, 0xf3ec98d6d2ca85c5ul, 0x67b7c8f6dcf27b5eul, 0xef67271196bca5b6ul, 0x9ed475cb50529436ul, 0x307a5cbcbb5b24c6ul, 0x8b84e9b9eecc599eul, 0xa694c57a6235afc0ul, 0x4d4bdc21ca678669ul, 0xf89fbdf824a6f80dul, 0xde4e502d5566064ul, 0xed105eb7e6e66179ul, 
   0xe489d65918050163ul, 0x5e1b6e18af9f5043ul, 0x6b50bb32820b92dful, 0xbe8ee858bf9596ceul, 0x3079053bff307936ul, 0x4507a42e47cd142ful, 0x914458b9a72711a9ul, 0xdafc4605029d0adbul, 0x8cd9b214030a4877ul, 0x58dd7e764659273eul, 0x565eace437cf1a16ul, 0x1ce0962a60b8491dul, 0x7692c497f5f7305ful, 0xfc2d2de1b95a8e4ul, 0xcbd4db4e72bb58eul, 0xf9a3d3fef9e9ae4ful, 
   0x93c060b266041f73ul, 0x315705b93177a7e7ul, 0xc87d9311d4c9db49ul, 0x8cefb7f50404cc29ul, 0x22b953eec91d64a1ul, 0x7db1aa5a92ef7b68ul, 0x6ff3114f64705aa9ul, 0x25525b9dd3909ff9ul, 0xcda8412d09ecd831ul, 0x5c158bbd94ace6bcul, 0xe2cdde3272de1d6aul, 0xccf42ae8a6fa8adul, 0xa6b89100686531d3ul, 0x6840fa7094e41d2ful, 0x397a8f2aafdf97faul, 0xcf66e2727052ecb0ul, 
   0x9a8aca86fcb72bd6ul, 0xcf2c78fd1eaa5c2ful, 0x43682dcc9c6fb5aeul, 0x6b72b172b8686ad3ul, 0xa785db0568d958f9ul, 0xe80c6efbb59aca4cul, 0x96026ac6d4f5ed92ul, 0x2f81ddc2fc45a123ul, 0xc6baf7e5fddc5f18ul, 0x46667b2f1b0b3b7dul, 0xc715b7a577eaafceul, 0x6455ae696633ec0ul, 0x7aba8efb6ab990d2ul, 0xb59192b52bbbced1ul, 0x81a0f95e03b37868ul, 0xd18b112523b37a92ul, 
   0xd60fc8a14b7139d6ul, 0x40923a9c022337a7ul, 0x8980000a80900731ul, 0x315aaad8cf476e7dul, 0xab23e097b9b2f5eeul, 0x8339e6b525e7a9b0ul, 0x3a535ce5a8bf2163ul, 0x2591eca393f80223ul, 0xd72743c3f2a4162eul, 0xf1e7646735d9b778ul, 0xb0984dc7716157b8ul, 0x244e59e0329718a6ul, 0xe20e2055f8961000ul, 0x7f2faf201127fb27ul, 0x64e418b57947f434ul, 0x533b20c72c76b1c3ul, 
   0x8e374130cc4d1512ul, 0x5730a9b6c7ed5b8ful, 0x9383f3d0f05417aeul, 0xeef75c7d61ed0511ul, 0x4ff928f476a03b76ul, 0x1e01157c7758bfc9ul, 0x53a96e988cb090bbul, 0x37f3dd82cfb704f4ul, 0xb0bb2a2a97d03baul, 0xf32a513ce832eb8aul, 0x9e7a4e90ffdbc152ul, 0xdba3eaab3601bb6cul, 0x3fc8cf43596b27f7ul, 0x6286aeb1ad91d531ul, 0x5296cc889a2091e6ul, 0x530976e2a28e62f3ul, 
   0x47e315d225613113ul, 0xe939d0e918837c25ul, 0x78bc544dbab2f53cul, 0x95c6b95a1d69cb7bul, 0xc66e8749bef57f5bul, 0x6dee3e9a492fba7dul, 0x2aa1be0a678b123dul, 0xa3a6792878c5058ful, 0x96883e8a684bb681ul, 0x749920fd64f4c3c4ul, 0x3c032c20886126a6ul, 0x918a395e4c9b38bbul, 0x71bc194d33b6a99ul, 0xec9617b375df67e7ul, 0x40d3f0b335b18f9ful, 0xfa0f0fb2798a4d6eul, 
   0x3f0c3312f2db1a5ul, 0xca0c0ad0645b9bd6ul, 0x20404036f9ee373eul, 0xa7c0314a66f03219ul, 0x9bf0ac28088831a7ul, 0xdf82ace9a159d168ul, 0x63e782f262ce9b41ul, 0xaf8a7be7962c10a9ul, 0x37ae5fa97cedace4ul, 0x3bbf1307e3facd8aul, 0x4d5ce5c5e2d2fab4ul, 0xdaec34bccdbd3353ul, 0x8043b95e3cbc5faeul, 0x70a13192c0c007ccul, 0x8c0b0c4769be1531ul, 0xa6c6cb0921909ab0ul, 
   0x2b51445adef1716bul, 0xcb55010aa8b952b9ul, 0x55a48a37576aaba2ul, 0xafbc85b9a3bba901ul, 0x93932817e292b366ul, 0xd260d688a651fd27ul, 0x627d0f0b8dc0ce1bul, 0x1d9301458e200440ul, 0xd13dcc3a316e438ul, 0x18a896a6238ac2eeul, 0x6a749f47885c94d1ul, 0x19e6f0c66cb7c8dbul, 0x5f7c923c4f77637dul, 0x98ca2e94e5e79810ul, 0xf20ae626b28798f7ul, 0x5ad91b933daf7d8ul, 
   0x408dbe6ea13d1e24ul, 0xf91f14b8ec8002c0ul, 0x752c1570b0304f09ul, 0xf8370ac27b822788ul, 0xfe53136514813d4ful, 0xe4bf758009de5f0bul, 0xdb6cf6d32df1dc2cul, 0xbaef6ad7b477badcul, 0x6e081a29a4b2392eul, 0xe36263b86f627477ul, 0x4c1487b09005d8d1ul, 0x18dc97a63ad86d8ful, 0xc2e09b250cdd3b22ul, 0xf889a5e9826cbf2aul, 0x83cdf1b4d4b70668ul, 0x10b6e59965718413ul, 
   0xc5cbce417ef82218ul, 0x136ba0b144d4569cul, 0x7964a2d1fd8dedc7ul, 0x3a59d0004f91eedeul, 0x97c52ca66fea60c0ul, 0x7241880886020221ul, 0x13f1f22eaebd2becul, 0x15bd2133e38858cdul, 0x1aeff080094ddd04ul, 0x482f1629bde1dafaul, 0x298c492dcd65a824ul, 0x83dc56143ea72d93ul, 0x2e5e9ed864ee991ul, 0x60d1eb8f2730a05ful, 0x445749c28185ea30ul, 0xcea7262febbf21f7ul, 
   0x5da9d0a8f8eee592ul, 0x4008f24fa7800416ul, 0x47fbd7833fd5fc1ul, 0xaa0abedbe330b884ul, 0xdca658b5dc384190ul, 0xdc08337641ef87cul, 0xc77a7b65f5c91082ul, 0xdecd13e63a31646ul, 0x1432d3e931448311ul, 0x6e266670dc1b989aul, 0xefbd9cce99992fc5ul, 0x1f3dccf0d4258223ul, 0xccdcb2bb858dfc51ul, 0x3901e279bbbbbd80ul, 0x6e7965a16c8e770ul, 0xeb8629cd10434cf1ul, 
   0xcf84758630185317ul, 0xc58abf1911532374ul, 0x8ddb8b5c745330eful, 0xde00102af1722a9dul, 0xaacc400037eadb1bul, 0x3075bd3f2099eee7ul, 0x659679e8611100f0ul, 0xabd1834c14040102ul, 0xb9dc04a0c0122eb9ul, 0x3d29f7c0822aa985ul, 0x12c2b7188624c6aaul, 0x447449c8ab53478eul, 0x8c90108d7d13fa93ul, 0xb8ed6008b8d9366dul, 0xac0882ae4a382f8bul, 0xd92f301328e203a1ul, 
   0x90022249e8022713ul, 0xa0554415773cfa17ul, 0xc9623983c51e9dd3ul, 0x83353a29a6e12c74ul, 0x17b56d041084d9fcul, 0x2bef416078004015ul, 0xe51be7bdf53e0a79ul, 0xd2bf2bf520302e2ul, 0xf47e0236c11040a0ul, 0x7a70a0c103e4aa82ul, 0x384497adf6a09509ul, 0xc06231969b8b0c75ul, 0x7cf223a9568a051ful, 0xb77803d738f9b7aaul, 0x6c14fc60008197b2ul, 0x98bb02dba78005faul, 
   0x22ad2e5f4239cad3ul, 0x6cb5a29d04164e63ul, 0xcec07465ecf80000ul, 0x42a572e5964c59c4ul, 0xa9c41837af57ca80ul, 0x5843953aab3e2103ul, 0xa9d2284e4c25488ful, 0x2f2b537daf4b4593ul, 0x6f505f0e57f4e699ul, 0xd8371c4d64e00021ul, 0x835802c5919e1c1eul, 0x4967663f5572d036ul, 0x7fd90222404f32edul, 0xa59a4fa2657b3b31ul, 0xa354d6ac5172bc8cul, 0xd2f2ead599c228acul, 
   0xbc7da5bfad9b9b8aul, 0x912044ee4845a94dul, 0x453a2811a7d2ceb0ul, 0xf2d5f55ec38335c3ul, 0x7988100d1c2fd7daul, 0x695b82020687a5eaul, 0x675bdbb1c3e593f8ul, 0xb17585603dadb04cul, 0x18f7cf41b5b5d61ul, 0xae773f822ea2e941ul, 0xcc8585851a146996ul, 0x95e599e6f58bf9faul, 0xcdad949ccaa5cbc2ul, 0x356e5d4ff8e8acaful, 0xb8670a455377a939ul, 0x8ff71c7c9c4a573ul, 
   0xb0cbe1777cd15f23ul, 0x4444dc5d7fb80805ul, 0x672d95bdb54d6b40ul, 0x7b45df7b319096c8ul, 0x2c37dc75e5d4e9b6ul, 0xed50ce4717a05901ul, 0xc14cb301b355ab8ful, 0x5556a542cae5e48cul, 0x372545b255de4172ul, 0x8b9824c9076c9939ul, 0xe44e5f1f14c4c8a1ul, 0x625bb72f1622ebe3ul, 0x8b13f2fce93cf15eul, 0xb34440263e97e121ul, 0x891d31f6790c905eul, 0xeeba964bb77244f3ul, 
   0xb645b5d7bb50a288ul, 0x614a8656cdd0abaul, 0x2514517c955d6601ul, 0xa28988339729444ful, 0xa3e58a8cced11791ul, 0xf145db8fa707efb7ul, 0x78496ffb7201ac2ful, 0x8477bd5e5fae7bf0ul, 0x985eaf0b343fa0b9ul, 0x880887592f0786f1ul, 0x82dc56b0de292f68ul, 0x77a121f554f59a7dul, 0x5c8ecac6d05d8d96ul, 0xbce6e454d64c8da9ul, 0x594d546ca13c5352ul, 0x13d299f04480e574ul, 
   0xbe6eaa48819cf00cul, 0xf0843144a39a61bcul, 0x3d611254c1a785e2ul, 0xed9c9832f3e2fc55ul, 0x572c77bf25676f43ul, 0xeaf6797dc7bbc9ceul, 0xc302209f4be399f5ul, 0x10e90d991996b949ul, 0x59809d6b2f15cb0eul, 0xaee975960b37f19eul, 0xb99115edf53361beul, 0xcb181689d4554e5eul, 0x524f230810b28a06ul, 0x7a0b2aab2832d5dul, 0x1d4e9e50dc547145ul, 0x33718efcde11009ful, 
   0x2152299e07c420d1ul, 0xcb8bf1ee8863033ful, 0x41fd89cf341ac79bul, 0x1463cce42b90c7ful, 0x4cd25b73575ac11ul, 0xc07518585496342dul, 0x37efc84d0d8fe2bcul, 0xcdd931c756bf0db3ul, 0x1aa1e5944b3c6402ul, 0xc98b4651301ba2a8ul, 0x55720414a62be0ul, 0xb5fa17495bdc9726ul, 0xc2004442747e4eeful, 0x12c7c64ab71ac2dbul, 0xa5dcfa7acc0720b7ul, 0x27d57e7b6127b87ul, 
   0x68807a71585a4432ul, 0xd05cfc93f14a1a54ul, 0x79ae9b4b0115ebb6ul, 0xd4699f64744ce0b6ul, 0xcb119389661929aful, 0xc14344e0cfcb758cul, 0x47be8527f2047cf1ul, 0x5296f81aa11a58e6ul, 0xd24e9bb86ee7fbbdul, 0x21dff7dc37eaful, 0x8984d584b7ebfda8ul, 0x4bf69df6f16a8f8eul, 0xef194bebebb9e3eeul, 0xa92042eb2cdcab51ul, 0xc95a300ba1058585ul, 0x6cd6fd96660fe464ul, 
   0xb9ce902b61528991ul, 0x5e66a96c9ec349e0ul, 0x6040a2bb7ccc343ul, 0xf5cf13f8020824c9ul, 0x1d743ec8fe98983cul, 0x17ffbd64068eef3bul, 0xd1deebe77deddc56ul, 0x972ff9bd800872fful, 0xf2b4f9bae41aa7f6ul, 0x7f57c285e5711010ul, 0x9c6ad7b8a361dce7ul, 0xad4572c0c5087499ul, 0x62f3aa1edf2e78b8ul, 0x364bb2f4df39bad9ul, 0xb48da73935b305c9ul, 0xd1d7ed58fec65680ul, 
   0xe81604278cd45619ul, 0xc68db619a4b260ccul, 0x98f7ec60c13a7558ul, 0x5b1ecb3fc7603077ul, 0xe4bfcb80004134acul, 0x7feffaa401dfdffbul, 0x97747d35616d7583ul, 0xf6fc108860a59d4eul, 0xae41f4fcefbb2ad7ul, 0x961a1c3e8159fc76ul, 0xac5eaf96820512a8ul, 0xd9d3e6daa517c93eul, 0x35b8b97c8ca9a4faul, 0xb08ae10adbea459ul, 0xaf75b38e010519aul, 0x5d65e7fcc448c02ul, 
   0xe3b1ccf6cf03b021ul, 0xe6a30563b1b0284ful, 0xf7ebebeb95e7650bul, 0x5bffdd1abbf2a001ul, 0x1686f42ddfee7a02ul, 0x816564a79da9ab09ul, 0xfedaf2bf57ba4345ul, 0xe689171bdaf2830ful, 0x1a33bb0f0bb31ecful, 0xaad2d41029144053ul, 0x4662c1c6e37a6a27ul, 0x7b927e5b200e545bul, 0x6c3403e33129d76eul, 0x4b0fd7d5a758c8b7ul, 0x22b2559fbb98f1dul, 0x378647111d26553dul, 
   0x3a349e58ad92fc0eul, 0x501c0f49b5354b66ul, 0x7825d97f63d6bb7eul, 0xa28593bc732f7e3ul, 0x412662e9d979fcabul, 0xfad67def27ee9ceaul, 0xe62cf54a00eec6fcul, 0xbecd667db8436cb3ul, 0x5dd749f0596c9597ul, 0x822c6f72b7ea295ful, 0xd27b9e1eacb1b3d7ul, 0x73b027f3eb3d156cul, 0x64ed6c993e3ec186ul, 0xb864255ac83d93b1ul, 0x37516c4f2bcf602bul, 0x5afab5bdf573cb17ul, 
   0x990fab7dd99ae0a9ul, 0xb8497783f648bc5ful, 0xa4932f4f93faebfcul, 0x1737222b8e4e2671ul, 0xeb69efc10c0f91c3ul, 0x57eac8cba076bdfdul, 0x30ceb98cf7c22a81ul, 0xcae376182c81bb27ul, 0x41e86e4183d1991bul, 0xd2b212347335bdd3ul, 0xf0ffa2739d8a0fdul, 0xef3f0ecf18c0edacul, 0xf8162dc3228b191eul, 0x98bef9b5d5f5b54bul, 0xdafeb40a1bfb7797ul, 0x9f17e7f217bbb40cul, 
   0x9aff7fe467be0147ul, 0x58ffb6026fddf372ul, 0xc48e3f1de306d64cul, 0x6760af5cd71e28e1ul, 0xff28e70472410517ul, 0x94cad9d97fde8fcdul, 0xff0c27369815c0c5ul, 0x7b3fc8f64924bb07ul, 0xbe5fb33307a366d0ul, 0x8ccec5c7c017995dul, 0x26f3ce7e3fc12545ul, 0xb922babfbf41b02aul, 0x9c4f3035a60b8ceful, 0xf1e3059ac39df9c7ul, 0x4144498b04000070ul, 0x2b9a1f7f39dd9a54ul, 
   0xfc356fd2ce83e220ul, 0xffdc3f700bfdee1ful, 0x57f7a4237a2dcb7ul, 0x498c162e6fd90000ul, 0xa4b75e4788ec78f7ul, 0x7021523ba33c919dul, 0xa07282fd27c796e6ul, 0xc2f8799915ccd4faul, 0xd061c5db55f9bddul, 0xbb6cf54eec30d3a7ul, 0x9bb59ce4e2e140d5ul, 0x4f82f39385525d5ul, 0x65b839b2cbc32810ul, 0xa5c6b23d5f757a69ul, 0x542b90c64f6f48caul, 0xeaefe5feff302954ul, 
   0x7d1fe63dfb567f7bul, 0xf7152d446fc6cd6dul, 0xb0a0201e71e25467ul, 0xf605e808cf9cae76ul, 0x5cb0e614c625b2c4ul, 0x8c937b306567240ul, 0xdc8ff6b04e7b58daul, 0x84ce9278195b44c1ul, 0x7edc65ecf1cec7a7ul, 0x3a69baad27065af2ul, 0xea2c22f55038c65aul, 0xdfeef2c9737ec7e2ul, 0x341a40cdba65ce1dul, 0x964f759b8188befdul, 0x83d7f56ff103179eul, 0x62e5c810bcbd1e0ful, 
   0x721fd4e91103ba7ul, 0x8cdcd9a7b4febceeul, 0x4b4ad3032cf0bd7aul, 0xae0acb4d4c3065ebul, 0xc0f935071c554b01ul, 0xebcb219166a1c592ul, 0x85142f4fda0e4f33ul, 0xcdd0291f9f98d747ul, 0x4056717e61fde5e7ul, 0x5def1fb63f333a3cul, 0xa3283e8a84d6027dul, 0xf8b0b1fbd95e78ecul, 0xbfd05ac2e77f033cul, 0xc74ffd4a941186feul, 0x752dbc4bea71898cul, 0xca5eb2bc40110622ul, 
   0x7c75c0e35b67a6c8ul, 0x3ecd19f9dd5a0ecful, 0x427b524ba92a3208ul, 0xde89fd3a4c6facb2ul, 0x9faa2f159913a968ul, 0x20aeba572beb65fdul, 0x67b7ded6b4b6ccddul, 0x6e01b3b9dedef184ul, 0xfaeb05f707763f3dul, 0xb776e6cc73ad7f61ul, 0xfabf1e37f3e60f56ul, 0xbaf93e73d1b2226bul, 0x9ab6861bfdeaaa3ful, 0xe5f95b824a5afe4dul, 0xa39a97cbf56cd41aul, 0x23c741a3fa760feul, 
   0xa07db6828aac8260ul, 0x31dcd5279fac4a55ul, 0xeb555c82071cf203ul, 0x95f207e326771918ul, 0x3762ab9a651d5ff5ul, 0xfe1378fc7fc067bul, 0x74300580f37f6ad7ul, 0x1d7703ab71fe5f75ul, 0xe9e0a3a7d64a28cbul, 0x2537fd7ceeec6c6ful, 0xb051400ec0b9c4f4ul, 0xf8fbecacc2c7d0f7ul, 0x35e33e6c293a290bul, 0x3f9f27d58577e525ul, 0x4273fc2f8277101aul, 0x85b676d66f2bf908ul, 
   0x1d4572c0f3017a1cul, 0x73730aea940f327dul, 0x3d43881e1a3ef901ul, 0xb62abcef7cb7efebul, 0xa4f14f345785f8feul, 0x55caceef4bcbff4ful, 0x37bb35697c5a33a7ul, 0x22405cfd5dd6ab3ul, 0x22240adf2d1748f6ul, 0xe4a4df8d813ae0a1ul, 0xcbd9dcc3ab84b84ful, 0xe559e3dcfe6cbf2ul, 0xcfd52a56404443e9ul, 0x78efaa6a36dd2aacul, 0xe4d295353ee898cul, 0x15a4e0a15965cbcbul, 
   0x7622aa2b1c352900ul, 0xc2ebb73be48f8584ul, 0xef7b5db9dd5ab14cul, 0xd7367e5af16e41ceul, 0x143c159ba0552a5ful, 0x5d747434533c104ul, 0xb7e1189049018e19ul, 0xadb90746639487d2ul, 0xfcf55dfa6cfb7e37ul, 0x149e93dd1928bd7aul, 0x33e4800898914c08ul, 0xa7cc049122c922b1ul, 0xd82575d5f9efa58eul, 0xa7ff0e7fd40ff13cul, 0x4bf368670a93d85bul, 0x22b0b37f83bc82f9ul, 
   0x2f4985f432b2e62ul, 0x2db8c9501033ce55ul, 0x8b2d99d4705a8a87ul, 0x7e3679f4c4c8a28ul, 0xe9d0788831dfdc23ul, 0xcecc8d62e97a519cul, 0x31d32b7998cf6b66ul, 0xed4000739ca10ffbul, 0xef2ebc993a827be5ul, 0xa5f87fef0b04184ful, 0xa8547a3f9b13dd9ful, 0xbc9874f764e29afful, 0xc2f236b4138fb33aul, 0x115faddcabae992ful, 0x8df275dc3ed15efdul, 0x4a0e87471b065646ul, 
   0x7eeeb2b0b1f14685ul, 0x80a21e9c004458f2ul, 0x6c90a93fb8045164ul, 0x86c73ede179f4d1ful, 0xadceaf35133b5b5bul, 0x1dd119eb08e88606ul, 0xbb6642dab5d6ba69ul, 0x24bbe6cd9fb907fbul, 0xb3b6c6d8ef83c9fful, 0x2db6ca459e592540ul, 0x7c6f7cf77bcccdd5ul, 0x82e5c5e1e0827443ul, 0xdbe39cdacb07e07bul, 0x3fcdcb94ab75b0b6ul, 0x9f57b66e3070f33ful, 0xfff85ebb5d3fedebul, 
   0xd39cf77347101f71ul, 0x444e454900000000ul, 0x6c90a93f826042aeul, 

};
const std::size_t data_inputImage_size = 9748;





