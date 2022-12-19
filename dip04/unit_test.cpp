//============================================================================
// Name        : main.cpp
// Author      : Ronny Haensch
// Version     : 2.0
// Copyright   : -
// Description : only calls processing and test routines
//============================================================================

#include "Dip4.h"

#include <iostream>
#include <random>

using namespace std;
using namespace cv;

extern const std::uint64_t data_inputImage[];
extern const std::size_t data_inputImage_size;

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
            if (!fastmathIsFinite(mat(r, c))){
                std::cout << "r, c = " << r << "" << c << std::endl;
                return false;
            }
    
    return true;
}

bool matrixIsFinite(const Mat_<std::complex<float>> &mat) {
    
    for (unsigned r = 0; r < mat.rows; r++)
        for (unsigned c = 0; c < mat.cols; c++) {
            if (!fastmathIsFinite(mat(r, c).real())){
                std::cout << "r, c = " << r << " " << c << std::endl;
                return false;
            }
            if (!fastmathIsFinite(mat(r, c).imag())){
                std::cout << "r, c = " << r << " " << c << std::endl;
                return false;
            }
        }
    
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

        //std::cout << "ref = " << std::endl << " "  << ref << std::endl << std::endl;
        
        Mat_<float> res = dip4::circShift(in, -1, -1);
        //std::cout << "res = " << std::endl << " "  << res << std::endl << std::endl;
        if (!matrixIsFinite(res)){
            cout << "ERROR: Dip4::circShift(): Inf/nan values in result!" << endl;
            return false;
        }

        if (sum((res == ref)).val[0]/255 != 9){
            cout << "ERROR: Dip4::circShift(): Result of circshift seems to be wrong!" << endl;
            return false;
        }
    }
    {
        cv::Mat_<float> in(30, 30);
        cv::randn(in, cv::Scalar(0.0f), cv::Scalar(1.0f));
        
        cv::Mat_<float> tmp;
        tmp = dip4::circShift(in, -5, -10);
        tmp = dip4::circShift(tmp, 10, -10);
        tmp = dip4::circShift(tmp, -5, 20);

        if (!matrixIsFinite(tmp)){
            cout << "ERROR: Dip4::circShift(): Inf/nan values in result!" << endl;
            return false;
        }

        if (sum(tmp != in).val[0] != 0){
            cout << "ERROR: Dip4::circShift(): Result of circshift seems to be wrong!" << endl;
            return false;
        }
    }
    return true;
}

bool test_DFTReal2Complex()
{   
    
    {
        std::vector<std::pair<unsigned, unsigned>> sizes = {
            {3, 3},
            {8, 15},
            {16, 7},
        };
        
        for (auto s : sizes) {
            Mat_<float> in(s.first, s.second);
            in.setTo(0.0f);
            
            Mat_<std::complex<float>> out = dip4::DFTReal2Complex(in);
            
            if (!matrixIsFinite(out)){
                cout << "ERROR: Dip4::DFTReal2Complex(): Inf/nan values in result!" << endl;
                return false;
            }

            if ((out.rows != in.rows) || (out.cols != in.cols)) {
                cout << "ERROR: Dip4::DFTReal2Complex(): wrong size returned." << endl;
                cout << "   Expected: " << in.rows << "x" << in.cols << endl;
                cout << "   Got     : " << out.rows << "x" << out.cols << endl;
                return false;
            }
        }
    }
    
    {
        cv::Mat_<float> in(16, 16);
        in << -26.7577f, -202.053f, 60.6618f, -26.4472f, 18.8104f, -185.479f, 63.5832f, -81.7188f, -68.1865f, 220.516f, 76.7747f, -83.699f, -219.984f, 238.088f, 33.4434f, 242.003f, 165.095f, 115.73f, 107.141f, -90.662f, -32.1055f, -25.5698f, 13.4688f, 14.2683f, 23.7129f, -6.40971f, -175.497f, -97.3215f, -107.975f, 11.1516f, 73.9036f, 239.811f, 12.702f, 168.268f, 29.0835f, -74.7801f, 39.8418f, 145.397f, -83.4814f, -149.612f, 86.3251f, 173.003f, 30.0304f, -10.1007f, -3.45637f, -27.9582f, 93.9007f, -23.2767f, 64.176f, -54.5951f, 96.2804f, -7.10401f, 61.6773f, -18.5841f, -159.387f, 48.3501f, -296.804f, -6.38707f, -138.182f, -66.8438f, 152.469f, -70.8688f, 34.6206f, 42.5012f, 169.808f, -71.1662f, -239.362f, 31.0782f, 74.8836f, -46.9088f, -63.7945f, -82.1851f, 100.809f, -127.041f, -16.8828f, -132.803f, -48.1579f, 249.983f, -48.0548f, 114.111f, 3.30729f, -123.952f, -74.4078f, -64.3304f, -25.7336f, -66.9437f, -95.4322f, -166.114f, 73.9747f, -66.0802f, 217.281f, -101.877f, 190.282f, -106.768f, -83.6194f, 102.038f, -23.8934f, -149.29f, -167.529f, 64.5867f, -1.5901f, -81.6987f, -64.0477f, -37.4205f, 173.789f, 9.77879f, -123.958f, 15.3998f, 277.962f, -86.3666f, 129.34f, -84.9827f, -105.826f, -162.379f, 46.0402f, -49.8953f, 20.509f, 104.91f, 101.885f, 7.91392f, -77.7259f, 38.723f, 230.537f, -202.706f, 127.249f, 174.625f, -68.0329f, -40.0392f, 140.808f, -105.953f, -31.3377f, 131.682f, 76.564f, -82.0819f, -87.3848f, 106.043f, -1.05856f, 14.0332f, 179.122f, -70.3709f, -23.4234f, 40.8061f, -88.4523f, -25.5001f, 0.873297f, 105.29f, 83.2829f, -50.0105f, 37.7918f, 66.9004f, -184.633f, -57.926f, -231.884f, 87.4483f, 27.3093f, 11.7091f, 49.3303f, 0.961097f, 45.0684f, -165.11f, -138.98f, -31.9308f, 108.23f, 49.0188f, -83.4698f, 55.1627f, 180.24f, -92.4778f, 82.5769f, -83.377f, 75.3906f, 194.647f, 70.0382f, 19.1312f, -2.28771f, -275.788f, 204.144f, 87.9446f, 15.2588f, 6.2551f, -327.731f, -245.763f, -19.2097f, 64.9429f, 156.631f, 75.3366f, 175.576f, 155.568f, 41.4057f, 35.1669f, 58.5925f, -70.6451f, 23.7217f, -113.833f, -27.3484f, -141.567f, 239.296f, 163.354f, 6.48436f, -23.9057f, -87.2819f, -103.787f, -174.097f, -126.505f, 105.515f, 32.979f, 223.442f, -70.5923f, -57.2261f, 93.761f, -56.9881f, -343.519f, 77.098f, 157.772f, 100.929f, 235.711f, 123.468f, 80.2486f, 119.384f, -29.6472f, -17.3404f, 15.2498f, 80.849f, -213.846f, -28.7173f, -59.3054f, 71.3961f, -155.017f, -176.339f, -125.405f, 49.0886f, 100.735f, 31.504f, -83.0471f, -14.5183f, 97.4379f, -317.139f, 168.216f, 125.713f, 56.3801f, -127.679f, -328.075f, 146.146f, 66.7579f, -69.2745f, 146.058f, 125.577f, -100.597f, -43.2341f, -177.949f, -9.28856f, 163.115f, 180.343f, 169.778f, 73.0169f, 140.535f;
        cv::Mat_<std::complex<float>> out_desired(16, 16);
        out_desired << 774.792f+0if, 133.925f+2892.81if, -715.902f+1548.54if, -872.562f+748.425if, -501.34f+-1223.11if, 644.372f+232.444if, 585.556f+1708.93if, 1010.02f+269.267if, 3065.96f+0if, 1010.02f+-269.267if, 585.556f+-1708.93if, 644.372f+-232.444if, -501.34f+1223.11if, -872.562f+-748.425if, -715.902f+-1548.54if, 133.925f+-2892.81if, 580.419f+1086.06if, 1565.07f+-1819.81if, 1181.43f+2197.2if, 1811.44f+988.984if, -874.041f+-2292.14if, 274.167f+-439.208if, 675.182f+1386.96if, -872.169f+3639.42if, -1607.98f+174.055if, 919.948f+-1421.2if, -498.993f+837.185if, -2901.3f+-1535.11if, -1764.07f+-2622.71if, -2146.81f+-689.905if, 1823.41f+-2093.75if, 1691.61f+-1469.18if, 784.984f+-708.585if, -694.247f+-2179.01if, -2645.32f+-1506.48if, -1174.79f+269.17if, -2925.93f+381.503if, 698.188f+1631.68if, -1102.82f+1025.67if, -261.794f+272.141if, -1410.44f+1169.47if, 104.541f+-2299.1if, 1358.13f+-724.653if, 1090.06f+-1003.35if, -914.237f+-481.676if, -146.895f+-1393.98if, 1776.93f+45.7625if, 1002.22f+-1485.15if, -20.9405f+-1083.57if, -614.436f+2261.35if, 799.674f+-1784.88if, -320.815f+250.001if, -1357.59f+711.305if, 27.4039f+2503.44if, 736.55f+496.736if, -2473.2f+-304.953if, 1125.9f+-1147.49if, -990.703f+-853.812if, 1914.04f+2373.96if, 232.917f+-1561.13if, 1339.14f+-2355.59if, -920.02f+120.279if, -669.978f+-1040.82if, 685.042f+-1255.03if, -99.827f+563.14if, 321.229f+-342.299if, -117.21f+-353.901if, 499.886f+1273.34if, -461.778f+-649.735if, 1031.61f+959.892if, -1070.71f+2584.37if, -2.63407f+946.589if, -91.3356f+-165.033if, -52.9982f+-639.648if, -368.321f+-749.183if, 4048.78f+1287.04if, 2549.79f+-997.728if, -670.475f+-2663.53if, 111.836f+-2853.04if, 2155.66f+563.971if, -448.297f+-410.523if, -863.899f+-75.9089if, -1487.94f+-1748.46if, -1192.42f+52.4621if, -266.326f+-136.824if, -1155.9f+-1490.74if, -845.877f+-1808.86if, 2486.78f+2012.42if, -18.6624f+754.402if, -161.143f+1217.37if, -2262.04f+-937.211if, 870.932f+-1588.55if, -239.237f+2137.01if, 460.408f+254.157if, -2147.66f+-3603.9if, -906.826f+1569.49if, 100.732f+1173.31if, -763.534f+-1292.51if, 2281.15f+-4329.29if, -370.87f+998.911if, 987.152f+267.216if, 405.51f+-818.278if, -1432.64f+134.247if, 537.639f+-1487.09if, -18.464f+1510.07if, -273.928f+-1525.06if, 590.27f+241.052if, 1640.37f+-94.441if, -2058.36f+-1549.86if, 230.943f+734.769if, 2032.84f+-6.65466if, -1970.71f+-3300.32if, -566.945f+1512.88if, -1008.67f+-253.84if, 1656.88f+415.881if, 89.199f+1034.31if, -711.658f+-730.265if, 1634.1f+-218.811if, 346.662f+1620.35if, 306.708f+243.63if, -855.434f+-721.688if, -690.482f+-1338.83if, -1422.88f+359.282if, -36.1033f+648.619if, -2138.4f+2761.23if, -204.082f+-1442.83if, 515.462f+2224.86if, 1189.23f+-1971.86if, -482.169f+0if, -299.242f+-533.9if, -22.6969f+1033.18if, -1598.71f+1754.17if, 847.602f+227.79if, 393.775f+-1080.55if, 1824.32f+-1655.53if, -929.866f+1512.08if, -238.953f+0if, -929.866f+-1512.08if, 1824.32f+1655.53if, 393.775f+1080.55if, 847.602f+-227.79if, -1598.71f+-1754.17if, -22.6969f+-1033.18if, -299.242f+533.9if, -566.945f+-1512.88if, 1189.23f+1971.86if, 515.462f+-2224.86if, -204.082f+1442.83if, -2138.4f+-2761.23if, -36.1033f+-648.619if, -1422.88f+-359.282if, -690.482f+1338.83if, -855.434f+721.688if, 306.708f+-243.63if, 346.662f+-1620.35if, 1634.1f+218.811if, -711.658f+730.265if, 89.199f+-1034.31if, 1656.88f+-415.881if, -1008.67f+253.84if, 100.732f+-1173.31if, -1970.71f+3300.32if, 2032.84f+6.65466if, 230.943f+-734.769if, -2058.36f+1549.86if, 1640.37f+94.441if, 590.27f+-241.052if, -273.928f+1525.06if, -18.464f+-1510.07if, 537.639f+1487.09if, -1432.64f+-134.247if, 405.51f+818.278if, 987.152f+-267.216if, -370.87f+-998.911if, 2281.15f+4329.29if, -763.534f+1292.51if, -448.297f+410.523if, -906.826f+-1569.49if, -2147.66f+3603.9if, 460.408f+-254.157if, -239.237f+-2137.01if, 870.932f+1588.55if, -2262.04f+937.211if, -161.143f+-1217.37if, -18.6624f+-754.402if, 2486.78f+-2012.42if, -845.877f+1808.86if, -1155.9f+1490.74if, -266.326f+136.824if, -1192.42f+-52.4621if, -1487.94f+1748.46if, -863.899f+75.9089if, -99.827f+-563.14if, 2155.66f+-563.971if, 111.836f+2853.04if, -670.475f+2663.53if, 2549.79f+997.728if, 4048.78f+-1287.04if, -368.321f+749.183if, -52.9982f+639.648if, -91.3356f+165.033if, -2.63407f+-946.589if, -1070.71f+-2584.37if, 1031.61f+-959.892if, -461.778f+649.735if, 499.886f+-1273.34if, -117.21f+353.901if, 321.229f+342.299if, -20.9405f+1083.57if, 685.042f+1255.03if, -669.978f+1040.82if, -920.02f+-120.279if, 1339.14f+2355.59if, 232.917f+1561.13if, 1914.04f+-2373.96if, -990.703f+853.812if, 1125.9f+1147.49if, -2473.2f+304.953if, 736.55f+-496.736if, 27.4039f+-2503.44if, -1357.59f+-711.305if, -320.815f+-250.001if, 799.674f+1784.88if, -614.436f+-2261.35if, 784.984f+708.585if, 1002.22f+1485.15if, 1776.93f+-45.7625if, -146.895f+1393.98if, -914.237f+481.676if, 1090.06f+1003.35if, 1358.13f+724.653if, 104.541f+2299.1if, -1410.44f+-1169.47if, -261.794f+-272.141if, -1102.82f+-1025.67if, 698.188f+-1631.68if, -2925.93f+-381.503if, -1174.79f+-269.17if, -2645.32f+1506.48if, -694.247f+2179.01if, 580.419f+-1086.06if, 1691.61f+1469.18if, 1823.41f+2093.75if, -2146.81f+689.905if, -1764.07f+2622.71if, -2901.3f+1535.11if, -498.993f+-837.185if, 919.948f+1421.2if, -1607.98f+-174.055if, -872.169f+-3639.42if, 675.182f+-1386.96if, 274.167f+439.208if, -874.041f+2292.14if, 1811.44f+-988.984if, 1181.43f+-2197.2if, 1565.07f+1819.81if;
        
        Mat_<std::complex<float>> out = dip4::DFTReal2Complex(in);
        
        if (!matrixIsFinite(out)){
            cout << "ERROR: Dip4::DFTReal2Complex(): Inf/nan values in result!" << endl;
            return false;
        }

        if ((out.rows != out_desired.rows) || (out.cols != out_desired.cols)) {
            cout << "ERROR: Dip4::DFTReal2Complex(): wrong size returned." << endl;
            cout << "   Expected: " << out_desired.rows << "x" << out_desired.cols << endl;
            cout << "   Got     : " << out.rows << "x" << out.cols << endl;
            return false;
        }
        
        if (sum(abs(out_desired - out)).val[0] > 16*16*0.05f){
            cout << "ERROR: Dip4::DFTReal2Complex(): Result seems to be wrong!" << endl;
            return false;
        }        
    }
    
    return true;
}

bool test_IDFTComplex2Real()
{   
    {
        std::vector<std::pair<unsigned, unsigned>> sizes = {
            {3, 3},
            {8, 15},
            {16, 7},
            {64, 64}
        };
        
        std::mt19937 rng(1234567);
        std::normal_distribution<float> dist(0.0f, 128.0f);
        
        for (auto s : sizes) {
            Mat_<float> in(s.first, s.second);
            for (unsigned r = 0; r < in.rows; r++)
                for (unsigned c = 0; c < in.cols; c++)
                    in(r, c) = dist(rng);
            
            Mat_<std::complex<float>> out = dip4::DFTReal2Complex(in);
            Mat_<float> out2 = dip4::IDFTComplex2Real(out);
            
            if (!matrixIsFinite(out2)){
                cout << "ERROR: Dip4::IDFTComplex2Real(): Inf/nan values in result!" << endl;
                return false;
            }
            
            if ((out2.rows != in.rows) || (out2.cols != in.cols)) {
                cout << "ERROR: Dip4::IDFTComplex2Real(): wrong size returned." << endl;
                cout << "   Expected: " << in.rows << "x" << in.cols << endl;
                cout << "   Got     : " << out2.rows << "x" << out2.cols << endl;
                return false;
            }
            
            if (sum(abs(out2 - in)).val[0] > 16*16*0.05f){
                cout << "ERROR: Dip4::IDFTComplex2Real(): Result seems to be wrong!" << endl;
                return false;
            }        
        }
    }
    
    return true;
}

bool test_computeInverseFilter()
{
    {
        std::vector<std::pair<unsigned, unsigned>> sizes = {
            {3, 3},
            {8, 15},
            {16, 7},
        };
        
        for (auto s : sizes) {
            Mat_<std::complex<float>> in(s.first, s.second);
            for (unsigned r = 0; r < in.rows; r++)
                for (unsigned c = 0; c < in.cols; c++)
                    in(r, c) = 0.0f;
            in(0, 0) = 1.0f;
            
            Mat_<std::complex<float>> out = dip4::computeInverseFilter(in, 0.01f);
            
            if (!matrixIsFinite(out)){
                cout << "ERROR: Dip4::computeInverseFilter(): Inf/nan values in result!" << endl;
                return false;
            }

            if ((out.rows != in.rows) || (out.cols != in.cols)) {
                cout << "ERROR: Dip4::computeInverseFilter(): wrong size returned." << endl;
                cout << "   Expected: " << in.rows << "x" << in.cols << endl;
                cout << "   Got     : " << out.rows << "x" << out.cols << endl;
                return false;
            }
        }
    }
    {
        std::mt19937 rng(1234567);
        std::normal_distribution<float> dist(0.0f, 1.0f);

        Mat_<std::complex<float>> in(32, 32);
        for (unsigned r = 0; r < in.rows; r++)
            for (unsigned c = 0; c < in.cols; c++)
                in(r, c) = std::complex<float>(dist(rng), dist(rng));
        in(0, 0) = 100.0f;
        in(1, 5) = 0.0f;
            
        Mat_<std::complex<float>> out = dip4::computeInverseFilter(in, 0.01f);
            
        if (!matrixIsFinite(out)){
            cout << "ERROR: Dip4::computeInverseFilter(): Inf/nan values in result!" << endl;
            return false;
        }
        for (unsigned r = 0; r < in.rows; r++)
            for (unsigned c = 0; c < in.cols; c++) {
                if (std::abs(in(r, c)) < 1.0f) {
                    if (std::abs(std::abs(out(r, c)) - 1.0f) > 0.01f) {
                        cout << "ERROR: Dip4::computeInverseFilter(): Result seems to be wrong for values < threshold!" << endl;
                        return false;
                    }
                } else {
                    auto v = in(r, c) * out(r, c);
                    if (std::abs(v - std::complex<float>(1.0f, 0.0f)) > 0.01f) {
                        cout << "ERROR: Dip4::computeInverseFilter(): Result seems to be wrong for values > threshold!" << endl;
                        return false;
                    }
                }
            }

    }
    return true;
}


bool test_applyFilter()
{
    {
        std::vector<std::pair<unsigned, unsigned>> sizes = {
            {3, 3},
            {8, 15},
            {16, 7},
        };
        
        std::mt19937 rng(1234567);
        std::normal_distribution<float> dist(0.0f, 1.0f);
        
        for (auto s : sizes) {
            Mat_<std::complex<float>> in(s.first, s.second);
            Mat_<std::complex<float>> in2(s.first, s.second);
            for (unsigned r = 0; r < in.rows; r++)
                for (unsigned c = 0; c < in.cols; c++) {
                    in(r, c) = std::complex<float>(dist(rng), dist(rng));
                    in2(r, c) = std::complex<float>(dist(rng), dist(rng));
                }

            Mat_<std::complex<float>> out = dip4::applyFilter(in, in2);
            
            if (!matrixIsFinite(out)){
                cout << "ERROR: Dip4::applyFilter(): Inf/nan values in result!" << endl;
                return false;
            }

            if ((out.rows != in.rows) || (out.cols != in.cols)) {
                cout << "ERROR: Dip4::applyFilter(): wrong size returned." << endl;
                cout << "   Expected: " << in.rows << "x" << in.cols << endl;
                cout << "   Got     : " << out.rows << "x" << out.cols << endl;
                return false;
            }
            
            for (unsigned r = 0; r < in.rows; r++)
                for (unsigned c = 0; c < in.cols; c++) {
                    auto v = out(r, c) / in(r, c);
                    if (std::abs(v - in2(r, c)) > 0.01f) {
                        cout << "ERROR: Dip4::applyFilter(): Result seems to be wrong!" << endl;
                        return false;
                    }                    
                }
            
        }
    }
    return true;
}


bool test_inverseFilter()
{
    cv::Mat img = cv::imdecode(cv::_InputArray((const char *)data_inputImage, data_inputImage_size), 0);
    img.convertTo(img, CV_32FC1);
    
    cv::Mat_<float> degraded, kernel;
    kernel = dip4::degradeImage(img, degraded, 2.0f, 100.0f);
    
    std::vector<std::pair<float, float>> tests = {
        {1.0f, 24.0895f},
        {0.1f, 26.6553f},
        {0.01f, 14.7256f},
    };
    
    for (auto t : tests) {
        cv::Mat_<float> filtered = dip4::inverseFilter(degraded, kernel, t.first);
        
        if (!matrixIsFinite(filtered)){
            cout << "ERROR: Dip4::inverseFilter(): Inf/nan values in result!" << endl;
            return false;
        }

        if ((filtered.rows != img.rows) || (filtered.cols != img.cols)) {
            cout << "ERROR: Dip4::inverseFilter(): wrong size returned." << endl;
            cout << "   Expected: " << img.rows << "x" << img.cols << endl;
            cout << "   Got     : " << filtered.rows << "x" << filtered.cols << endl;
            return false;
        }

        
        cv::Mat_<float> diff = filtered - img;
        float meanSqrDiff = cv::mean(diff.mul(diff))[0];
        float PSNR = 10.0f * std::log10(255*255 / meanSqrDiff);
        
        if (std::abs(PSNR - t.second) > 0.01f) {
            cout << "ERROR: Dip4::inverseFilter(): Result seems to be wrong!" << endl;
            cout << "   with eps " << t.first << " expected a PSNR of " << t.second << " but achieved a PSNR of " << PSNR << endl;
            return false;
        }
    }
    return true;
}

bool test_computeWienerFilter()
{
    {
        std::vector<std::pair<unsigned, unsigned>> sizes = {
            {3, 3},
            {8, 15},
            {16, 7},
        };
        
        for (auto s : sizes) {
            Mat_<std::complex<float>> in(s.first, s.second);
            for (unsigned r = 0; r < in.rows; r++)
                for (unsigned c = 0; c < in.cols; c++)
                    in(r, c) = 0.0f;
            in(0, 0) = 1.0f;
            
            Mat_<std::complex<float>> out = dip4::computeWienerFilter(in, 10.0f);
            
            if (!matrixIsFinite(out)){
                cout << "ERROR: Dip4::computeWienerFilter(): Inf/nan values in result!" << endl;
                return false;
            }

            if ((out.rows != in.rows) || (out.cols != in.cols)) {
                cout << "ERROR: Dip4::computeWienerFilter(): wrong size returned." << endl;
                cout << "   Expected: " << in.rows << "x" << in.cols << endl;
                cout << "   Got     : " << out.rows << "x" << out.cols << endl;
                return false;
            }
        }
    }
    {
        std::mt19937 rng(1234567);
        std::normal_distribution<float> dist(0.0f, 1.0f);

        Mat_<std::complex<float>> in(32, 32);
        for (unsigned r = 0; r < in.rows; r++)
            for (unsigned c = 0; c < in.cols; c++)
                in(r, c) = std::complex<float>(dist(rng), dist(rng));

        {
            Mat_<std::complex<float>> out = dip4::computeWienerFilter(in, 1e5f);
                
            if (!matrixIsFinite(out)){
                cout << "ERROR: Dip4::computeWienerFilter(): Inf/nan values in result!" << endl;
                return false;
            }
            for (unsigned r = 0; r < in.rows; r++)
                for (unsigned c = 0; c < in.cols; c++) {
                    auto v = in(r, c) * out(r, c);
                    if (std::abs(v - std::complex<float>(1.0f, 0.0f)) > 0.01f) {
                        std::cout << "r, c = " << r << " " << c << std::endl;
                        cout << "ERROR: Dip4::computeWienerFilter(): Result seems to be wrong for very high SNRs" << endl;
                        cout << "     expected in each cell " << std::complex<float>(1.0f, 0.0f) << " but got " << v << endl;
                        return false;
                    }
                }
        }
        {
            Mat_<std::complex<float>> out = dip4::computeWienerFilter(in, 1e-5f);
                
            if (!matrixIsFinite(out)){
                cout << "ERROR: Dip4::computeWienerFilter(): Inf/nan values in result!" << endl;
                return false;
            }
            for (unsigned r = 0; r < in.rows; r++)
                for (unsigned c = 0; c < in.cols; c++) {
                    auto v = in(r, c) * out(r, c);
                    float expected = std::norm(in(r, c)) * 1e-10f;
                    if ((std::abs(v.real() - expected) > 1e-12f) || (std::abs(v.imag()) > 1e-12f)) {
                        cout << "ERROR: Dip4::computeWienerFilter(): Result seems to be wrong for very low SNRs" << endl;
                        cout << "     expected in cell " << std::complex<float>(expected, 0.0f) << " but got " << v << endl;
                        return false;
                    }
                }
        }
    }
    return true;
}

bool test_wienerFilter()
{
    cv::Mat img = cv::imdecode(cv::_InputArray((const char *)data_inputImage, data_inputImage_size), 0);
    img.convertTo(img, CV_32FC1);
    
    cv::Mat_<float> degraded, kernel;
    
    std::vector<std::pair<float, float>> tests = {
        {10.0f, 26.4129f},
        {50.0f, 27.5213f},
        {75.0f, 27.1179f},
        {100.0f, 26.0791f},
        {125.0f, 24.9601f},
        {150.0f, 23.6839f},
        {200.0f, 21.1568f},
    };
    
    for (auto t : tests) {
        kernel = dip4::degradeImage(img, degraded, 2.0f, 100);
        cv::Mat_<float> filtered = dip4::wienerFilter(degraded, kernel, t.first);
        
        if (!matrixIsFinite(filtered)){
            cout << "ERROR: Dip4::wienerFilter(): Inf/nan values in result!" << endl;
            return false;
        }

        if ((filtered.rows != img.rows) || (filtered.cols != img.cols)) {
            cout << "ERROR: Dip4::wienerFilter(): wrong size returned." << endl;
            cout << "   Expected: " << img.rows << "x" << img.cols << endl;
            cout << "   Got     : " << filtered.rows << "x" << filtered.cols << endl;
            return false;
        }

        
        cv::Mat_<float> diff = filtered - img;
        float meanSqrDiff = cv::mean(diff.mul(diff))[0];
        float PSNR = 10.0f * std::log10(255*255 / meanSqrDiff);
        
        if (std::abs(PSNR - t.second) > 0.01f) {
            cout << "ERROR: Dip4::wienerFilter(): Result seems to be wrong!" << endl;
            cout << "   with snr " << t.first << " expected a PSNR of " << t.second << " but achieved a PSNR of " << PSNR << endl;
            return false;
        }
    }
    return true;
}

// usage: path to image in argv[1], SNR in argv[2], stddev of Gaussian blur in argv[3]
// main function. Loads the image, calls test and processing routines, records processing times
int main(int argc, char** argv) {
    bool ok = true;
    
    ok &= test_DFTReal2Complex();
    ok &= test_IDFTComplex2Real();
    ok &= test_circShift();
    ok &= test_applyFilter();
    ok &= test_computeInverseFilter();
    ok &= test_inverseFilter();
    ok &= test_computeWienerFilter();
    ok &= test_wienerFilter();
    
    if (!ok) {
        cout << "There are still errors!" << endl;
        return -1;
    } else {
        cout << "No errors detected." << endl;
        return 0;
    }
} 



const std::uint64_t data_inputImage[] = {
   0x464a1000e0ffd8fful, 0x5e01010101004649ul, 0x4300dbff00005e01ul, 0x201010101010200ul, 0x202020202010101ul, 0x405020202020304ul, 0x606060506040304ul, 0x608090706060605ul, 0x80b080606070907ul, 0x8060a0a0a0a0a09ul, 0xa0a090c0a0b0c0bul, 0x8800080b00c2ff0aul, 0xc4ff00110101a000ul, 0x101030200001d00ul, 0x10101ul, 0x706030504000000ul, 
   0xdaff0900010208ul, 0xab01000000010108ul, 0xe3425dcf8e65997aul, 0x7e573babd5eda629ul, 0xf504b6594b5c9201ul, 0xd0907769549de8c2ul, 0xd26d37261143491aul, 0xa332c9809912324ful, 0xf14468574ddc8d69ul, 0x240418eef497b0ul, 0xef0db315fe16b24cul, 0xe8284176f8a723e9ul, 0x335f0b055b57ea3eul, 0xfc3fa61a877804e9ul, 0x9bddf5d3537ed457ul, 0x323e39f2a25eba1eul, 
   0x7ae8cf1659cb4d0cul, 0xb52ffbfaede996c1ul, 0x8008924bd863a45ful, 0xf3eaa2cfc28442d5ul, 0xf3ce5f5efa59f7b1ul, 0x9dd1cc9588c393fbul, 0x53650972fa4d6ccful, 0x6e81160b430057e4ul, 0x8f970ad1a1decc37ul, 0x785740d7fc5666eul, 0x7975bf972f0de914ul, 0xb024067e5fbf72a1ul, 0x1e3bc057d37849e8ul, 0x36f6b1de7336fe8ul, 0x5b4b59f95adc7a98ul, 0x9dfddb6c33740e0aul, 
   0x1bced56d562778cful, 0xaf74a354554726eeul, 0x99bc2a6ae9a17b32ul, 0xfefbfc5199fee84dul, 0x9fe856b2a9a4b288ul, 0x952e6cdbbacc885dul, 0xc23b6207ddaaea7bul, 0x378b6f9bdf8a7664ul, 0xe35426c198b293a4ul, 0xba54c1079bc225c3ul, 0x23373a640e8b6f82ul, 0x102700c4ff8f0795ul, 0x202020200030200ul, 0x30202ul, 0x706000502010403ul, 0x1622211514121113ul, 
   0x800daff33322423ul, 0x4694020501000101ul, 0xa2c63f87beacaa8aul, 0x16295ccdc05d3eb4ul, 0x9a9046f2df92ca8cul, 0x69db531a9d2b8265ul, 0xb85ae560dcded0cul, 0x88829c0a65986dd6ul, 0x557202ea56f2b165ul, 0x263ccaaae8259315ul, 0x5dec6de3c13ebb84ul, 0xa4218dc090a0bb4eul, 0x30d453d0f0323aa5ul, 0xac617cbe21d2b0d4ul, 0xf7127a191580ac42ul, 0x4d83f89a6a653c76ul, 
   0x53dfafca7643f277ul, 0xa8ecd6b43c824f25ul, 0xaa20667090c2ca44ul, 0x8d33ae63e9c59481ul, 0x2a1266e6529895a3ul, 0x71f8ed4f96524ba2ul, 0xb9fed2b02a44c66ful, 0x61c8aca1dd12b0acul, 0xcbabaa25d2a012c3ul, 0xa4a7157b2241b550ul, 0x42416cb8a60a0a6bul, 0x17c845a70fd5abe5ul, 0xff1a3f71bd3f902ful, 0x130e6bc335588300ul, 0xab6cb2880514f930ul, 0x45f023a22c6d797ful, 
   0xb0051c8fbdcf3a63ul, 0x2874a22890ca96d8ul, 0x20d7d7a9192da5d7ul, 0x5c700d444a19e3bful, 0xa72899367dfb1fb6ul, 0x3b37af589f93ab9eul, 0xfc6d4ac1c0407b1cul, 0x76bee0ae9c823df0ul, 0xf585793f8dd7543bul, 0x3731e702dbab5545ul, 0x9867aed7426ff507ul, 0xe96c76c8bbf85e1ful, 0xc0c79f23524a2d56ul, 0x330fa8941a5a132dul, 0xcdeb609dddfe1dcdul, 0xd0ccd93fc8c5fd9bul, 
   0xb29f0ea29d98977aul, 0x98413bb70f6daeb1ul, 0x8ca9896f9a1c461eul, 0xdc5cf7418ca96926ul, 0xd15649500ee89aa4ul, 0xcd451f4da5f88968ul, 0x576fb08b164d2deaul, 0x684de6d44f54f95ul, 0xd36e16bf7e94db94ul, 0xdb599f36dc5ab80cul, 0xd8cfdabafb87a6f8ul, 0xf160dbc3086c5440ul, 0xc92d6fbcf246a4b1ul, 0x4c581f439d00ff07ul, 0xd7f07bd9f3ca1e71ul, 0x5af7ceb7bd9537f9ul, 
   0x33cad11e4cacedb3ul, 0xa348c8e09919e158ul, 0x476a18cde92e2966ul, 0xe7e947e02c6a11c5ul, 0x1464b8d3c9cf5dul, 0x9961594667f151c9ul, 0x68453d15f9d4d281ul, 0x49c7bfa9a0bcb176ul, 0xb366fbeb56728dfcul, 0xde626f00afc57043ul, 0x6dfcf59a5fa1edbful, 0x22672cf6ae53fbadul, 0x88edcf0cea1600fful, 0xfdaf2257f377f809ul, 0xb5daac56a48dd606ul, 0x2f21cf2909b5665aul, 
   0xbe755e00ff42f50cul, 0x5bfabc0c62b5a3bful, 0x17c1a271e6156de2ul, 0x77a6f1f47dd0aacdul, 0xc5545e24cdebf5e0ul, 0xf73f5d9f74fdbb32ul, 0xc74f6a15496d66adul, 0x6caea7fce25bbc68ul, 0x276ed1c5681a7f53ul, 0xc17cd787df18c6e3ul, 0x35e96bc06c7916ddul, 0x32d3bdad05483cd2ul, 0x59ea0d3c2401c299ul, 0x93ebd97a5dd088ebul, 0x58938cf46b2a8d9dul, 0x3d1654b0e3b7326ful, 
   0xd151717d8c1a8a05ul, 0x25552bf70bb76246ul, 0x91fed8b8b9dca9ecul, 0x99ab36773d0b3bddul, 0xd665a84b4b07faf6ul, 0x903ce42962fb7b6aul, 0x1fb38534faaee67ul, 0x59655cc05a693c63ul, 0xde32a9a6b80ed69eul, 0x6673b3b974d70972ul, 0xc5ef35144073555cul, 0x2b9a15b40db3682ful, 0xce443421864203abul, 0xd30fc0f1d6233ca7ul, 0x78e277141d39e675ul, 0xff80c53c4f88dful, 
   0x1dc45d7316e34bfdul, 0xeccb9d75493bbf42ul, 0x18da86b6ca1e158bul, 0x5d6a6bd2e8b8db80ul, 0x5200b2ba473961adul, 0x4ea3c066d6385bd6ul, 0xbdd330bbf41ad1feul, 0x9bbfea4339bad73ul, 0xbdafb495f8c3a0bul, 0xb36f1e7494e73687ul, 0xbc4ba315fc28b650ul, 0xeac8c5e29d6a12a8ul, 0xf20cc08e766968ecul, 0xf2fb292da12f27caul, 0xbaa1ab405fd005d2ul, 0xbe882ff2e576e134ul, 
   0x5e6a49e909d082a0ul, 0x6863496479bd964bul, 0xbe68161a9aa63e61ul, 0xeea933736454f14cul, 0x8bf773f6318d177eul, 0x2f57d1d7440bf4f9ul, 0x69b72748fbd839c5ul, 0xe831454f835f3cb6ul, 0x5ede730b9c3eab75ul, 0x6819edae67a4cbaeul, 0x2a846dfd1a07eec0ul, 0xd087c2c99a8fb70eul, 0x320fc8c9dfdda0e9ul, 0x97f56cec1d8eebd1ul, 0xb02e6a89d2408884ul, 0x2557589dd60b2aa8ul, 
   0xd010ade7d63302f2ul, 0x60d6206dfd5cab82ul, 0x2d953ef294b0ca3eul, 0xaf1d8755b70ced8dul, 0x60eb45cd613b5335ul, 0x2fe1c00b94fdc590ul, 0x1962cdae42d0befdul, 0x8061998a4d284686ul, 0x3d6b99c9ba21d012ul, 0xff93d9e2a6403013ul, 0x2010200103d00c4ul, 0x109030605050304ul, 0x3020100000001ul, 0x2213053121120411ul, 0x8142147161514132ul, 0xf0d1b1522306a191ul, 
   0x4334241510e1c133ul, 0xdaff8272f1c26253ul, 0x23f060001010800ul, 0xebf727b74c019712ul, 0x6d4366946e25c452ul, 0x5becc622e92b2caaul, 0x2b7b5e16c352f8f0ul, 0xf1bed6df6f5b5b31ul, 0x60a2a8767e0dcb7ful, 0x436caa6ddbc46c2ful, 0x2233b5efb8f6d203ul, 0x5a2213eafdc58f15ul, 0x2d5cf814ee16e6c4ul, 0xa33a216b00ff1beeul, 0x8d75a4563bc95854ul, 0x7f24ed141e354cb2ul, 
   0x95a5fc5ce932ae78ul, 0xe15126232ead5b5bul, 0x97829dba7b2f9245ul, 0x15737db76688b90aul, 0x7bd01553d4dac87aul, 0xb3e7f044109fa6b5ul, 0x9443280422c4a106ul, 0xe795d11a43ed1b1ful, 0x93c63530b3bb736ul, 0x761c8b5404362c55ul, 0xb28bb4852e92fbful, 0xcedb29c9745ba6a1ul, 0x85bf5f7991d3c989ul, 0x2d227e6a6b548d65ul, 0x4fc4d0f9d3a7ef9dul, 0xe0d2fa960f426a27ul, 
   0x735e5841bf42549ful, 0x58c2a9f52920b68eul, 0xc04c12ee683d6da8ul, 0xb7c0f332282c4935ul, 0x60a9f4b2a837c185ul, 0x38ec162256771b49ul, 0xc14c7d2c0f2bbcb5ul, 0x2c152ebe52c85f10ul, 0xd115c3c0a7d2bb5bul, 0xf6c3e3486ad51883ul, 0x58fb00ffbe612d23ul, 0xec3095ab727a3978ul, 0x80779140acf4959bul, 0x77fa96692f316ae4ul, 0x3373d2c9701c49e9ul, 0x16170050a334a001ul, 
   0xbaec7a4cbcfb8a3bul, 0x54d450bf112a0f8aul, 0xcc16d0f9939a4fc1ul, 0x5c8afdd17e0d2902ul, 0x9f4acb5b58785159ul, 0x50e32fd0c99b5812ul, 0xba9eb71cb8fc5c16ul, 0x3e1c49cdb01b84d3ul, 0xe01a7da3d92b6ae3ul, 0x8b2c7d38d4fae9e9ul, 0xb366b32751334f9ful, 0x535a5f774be94510ul, 0xcf7ad7731f9bfb84ul, 0x61a501b0f8a856dul, 0x1d5b06752057d8d8ul, 0xad1ff27157cc956bul, 
   0xbf226935f5d78c73ul, 0x72a527a7ac15669dul, 0xba1f4f61ff511f1ul, 0xd3e8f4e66647c63bul, 0xacf0596d10bace07ul, 0x9ae028fd569b0d96ul, 0xdfb1d1f815d8aac0ul, 0x2eacda77488bda5bul, 0xb02f2a8ba3b1a92ful, 0x396d2e09ab3a7a0ul, 0xf07515dede5853ccul, 0xe1d078dbacf2c9bdul, 0x2383ca0fdbfe1ef8ul, 0x8d7f6b7ded5ac75aul, 0xce1a9f9a7018ad69ul, 0xa5e2c7a7b74eece6ul, 
   0x6642217fb6a3c611ul, 0x18d1f0d55bb6550eul, 0x17e6eccde86d80cful, 0x708b2757b3cb25f7ul, 0x185abf1dbed8da12ul, 0x8aada3b3066b08bcul, 0xae5abdf63deb0a9ful, 0x957e5da3469dd5ddul, 0xad2da2563e37fe72ul, 0xdb6115e8f13295feul, 0x926caef364f6d10cul, 0xd57a3359e7c34256ul, 0xdfe6b3d2b7defab3ul, 0xca20a98cd637b07aul, 0x77d66b53db45c3bful, 0xe44ab77d07527580ul, 
   0xfcb8d4b2e3109148ul, 0x3d2c759e8e9de800ul, 0x35b7adf4971f83e1ul, 0x7beba0acf6769f9cul, 0x5b645ddf69bbb3daul, 0x6bfc5ba79c82e95dul, 0xd55beb00ffc5a302ul, 0x5cadbc8d59b5b79aul, 0xa163f5de68d73ba6ul, 0x9dc73ab21f2946acul, 0x589650b3ef2d502cul, 0x33a0f0ef01995dfbul, 0x3a62bae8aa944e76ul, 0x9c3c7c2915351d73ul, 0x85b7a49749728986ul, 0xbc5bd4b4daa8ed72ul, 
   0x6f8db29ac4de7953ul, 0x487d97578fa54ae1ul, 0x9785951ec885b5b1ul, 0x76355a6d84849ac1ul, 0x9e5b3adf5e34f2b4ul, 0x4c0826d5603536faul, 0xcb43edfb960d656cul, 0xab81b791f225e57aul, 0x9d178e78d9ab7beeul, 0x6029fc1ab150691ful, 0x2c5d2e5b2215c4c1ul, 0xf359eb933bb63e2bul, 0x36b3dab091aed96dul, 0xfc2bfd7e2649cbddul, 0xbb51ebbbc6c5b531ul, 0x3ba056233f4eb2a1ul, 
   0x2f09957fb6ed4fd7ul, 0x91f4f695aed5cbdul, 0x6f01b3ddad31e7c2ul, 0xd8a0d2e112577b9dul, 0x541b5c5d950f84beul, 0xa0a689be5ad7dc49ul, 0xd47a3a19327cd57bul, 0xba5e83f721037fadul, 0xab52dbb6b20a7f2ful, 0x62bdf34987d263eeul, 0xf487fe9ead9aca70ul, 0x998b62100397c9feul, 0xc3cd49ada78735b3ul, 0xb6235925d9441bc9ul, 0x33f9ea2d6abf056ul, 0x32ea7c9415fe4671ul, 
   0x6d779b5cece4fdf0ul, 0x9a6447eaa2c4a9fcul, 0x70e2c185c2803f5eul, 0x903f744877c4c870ul, 0xa77650c3fa371fa8ul, 0xcfb8e451036f8331ul, 0xb130a97ced669835ul, 0xcd824b61970d0b33ul, 0xa7f4d257aa2e499eul, 0x31d7b206ed76d6c0ul, 0xd54509c42a6fcb57ul, 0xf21415b85a6f71e1ul, 0x6cc47f2b13c676c7ul, 0xe66bf8661f3abfbful, 0x296cc483446c20cbul, 0x283d579a653d1c30ul, 
   0x3237d63ec44b939aul, 0x6bfc7d03fd12c64dul, 0xc906860f77f8c311ul, 0x95a2a97a611c239aul, 0x3e62d44610247448ul, 0x622fb582414acf0ul, 0xadafd3cd67e50a3ful, 0xdd85854b33207063ul, 0x7d905111fa77dd74ul, 0xcedfe55470af4ee1ul, 0x531b3bb8817a24adul, 0x7b75b77669da1cc5ul, 0xefa72f1d7290b85cul, 0x95a8ef663a2953faul, 0x6eefe9239da7e43bul, 0xc22757cc6e259aeaul, 
   0x8b2b0669fb0ca9bul, 0xa9b5374aab5170d9ul, 0x8aec91246974f870ul, 0xff748cad60862637ul, 0x30a19047fd2f1500ul, 0x1d3a6656d0df1590ul, 0x94333b060dc2fbabul, 0x6b58e32ff48665b7ul, 0x3c7a6ba06e85e68bul, 0x589fb5105cc45eb8ul, 0xbd06ded6c5d968d4ul, 0x6cc29aa72feeecaful, 0xbedee31e1c780313ul, 0x2153a3dd32371615ul, 0x658d5f00ffbb85beul, 0xf51f0349eb584a5cul, 
   0xe190a2b6c57d1333ul, 0xdb446932b3452a25ul, 0x9d9e2a52d4f9fb7eul, 0x8e598c67afefef37ul, 0xbe89b7068db1caa7ul, 0x6d4293b51162d406ul, 0x3a2c8e34731c7152ul, 0x1aa6f8685fb1e646ul, 0x4d530d6b387946d2ul, 0x7bac1e93284de54ful, 0x28d138e1e78aba4cul, 0xfe5f1c278a82d67dul, 0x398169dac2b5fe72ul, 0x437db735fc96be25ul, 0xba12969064aef01bul, 0xd048e9fdee377283ul, 
   0x62e5b1dffa1ef944ul, 0xead89de03adcf069ul, 0x60e7e1c41ad4ef08ul, 0xff3aefb1769b5ful, 0xf2a0eaac2d765c3aul, 0x3652f15b8480c364ul, 0xfa4864ea7398823ful, 0x2f9c981c93daf148ul, 0xd3f96b76b9dc1629ul, 0x1cd17ccd1a2cb060ul, 0xa2a32bfcfad81b84ul, 0x4c7854faae64f7edul, 0x45dfcb461bea1e1eul, 0x8ddf5fb88d1e8bf1ul, 0x2668bf73c0c38749ul, 0xefcc1c8c6399a8f4ul, 
   0x4164f2d2b16028dcul, 0x81a1487c8ea6f070ul, 0x834651ddf262e4f6ul, 0xcab9356275d5fc4eul, 0x9986b98cde1f8ca2ul, 0xf2629ce1ced5e652ul, 0x7c0c922a7fd9d6eful, 0xd2f66aa71a0fdfe5ul, 0xfcd4bae2c9d2faf6ul, 0x979c80885138111bul, 0x1873c328bdf4afdcul, 0x2bee58eba3ae6d46ul, 0x3aee08836673c286ul, 0x71e273102b86da74ul, 0xbc14005ddbeca872ul, 0x8f24dc6116c5193ful, 
   0x74e052662f46e3c3ul, 0x1adef8eabc8dd8b2ul, 0xe80d1b9f1edf2b64ul, 0x1efc979754282cf1ul, 0xa0ae50e92dfa9875ul, 0xc810631097e2e0ebul, 0x5882ba9adcb0c39ful, 0xac1db4feb5f103f5ul, 0x2ac6691298cd3133ul, 0xfbc9a317d99e0426ul, 0x872306a3f4db77cbul, 0x24687d1d7afb2937ul, 0xf87611aa4221ca32ul, 0x106bfcb32f71281dul, 0x5bdb8b2417de6439ul, 0x6ecb3225ae71b0d4ul, 
   0x37bd75bcbd55a198ul, 0xd66658319998e01ful, 0x119ff764e47d6b7eul, 0x79a4f1488583884dul, 0x459ddf1f8fec36e4ul, 0xd296a298b2e1a2c6ul, 0x27831433d7f4944bul, 0xc932a5fd3a9f2baul, 0x884985cf5a3b489cul, 0xa90a739cd4969f62ul, 0x9f82f94ab0eab11cul, 0xa9f05fdd689e551bul, 0x3914ab123162c3a7ul, 0x393618b9ca2f3b58ul, 0x82de7a3a76dd00fful, 0xeb6b0f966b1621b6ul, 
   0xf2bf047b9194fef3ul, 0xb11c73d48e6f6f19ul, 0x3314cb7a839813d8ul, 0x7fef742b4aaf9803ul, 0x77491851181e2ea5ul, 0xa53700ff9f6e7196ul, 0x1066c1dced31f1c3ul, 0x4374c4213d413f6ul, 0xe9afbaa6dcaea99ful, 0x406da0df85136c4bul, 0x96d77ea7c3301636ul, 0x85955ec7043746d4ul, 0x23e1871143e727f5ul, 0xa6f1480fad574c34ul, 0x642abd5c5e587049ul, 0xbdae6a0c1b55feeaul, 
   0x98ac08506f6b349cul, 0xcb8900ff488ee479ul, 0x49c3884153c4bf41ul, 0x42bea5526c2e0738ul, 0x20ab03268e825196ul, 0xdac654002f6eaeeeul, 0xb753e1b7a85a6d68ul, 0xe5d18dd948957fadul, 0x5f899fc42a3250d4ul, 0xcc6ea93cc4274cb5ul, 0x5d15495680de5e92ul, 0x83a3f2acf41fe77ful, 0x465286a08085f47ful, 0xecbd72d2757e83e1ul, 0xd43b0ee1395af7caul, 0x3f1e1320cb5f23ful, 
   0xc4a3ff6723fdd98ul, 0xe5db0ad9bd5d7e61ul, 0x347600cdcb537bbdul, 0xbb6db90a2bd7f8faul, 0xb3639658cc14e965ul, 0x64692a16a1f12f02ul, 0x5f1f5f28c40c6d12ul, 0x40d996919c30599dul, 0x6c38874d693a9eb6ul, 0x8548aec1a56c87b9ul, 0x7bcd8d8717c34fd6ul, 0x7232ca21e28a2756ul, 0x92610c8bd2b765e4ul, 0xa7ef96beb5dded34ul, 0x6864bcf0843861c5ul, 0xd9f0b2947eb748b9ul, 
   0xa336036bf49c483ul, 0xb0fb290bc51600fful, 0x9b3957197b8d4fc3ul, 0xc4ff3a9fda7c77ul, 0x301020200011025ul, 0x1010304ul, 0x4131002111010000ul, 0xc1b1a19181716151ul, 0x800daffd1f1f0e1ul, 0xa51213f01000101ul, 0x2156616b5f50134cul, 0x292e990227066974ul, 0x9a9237e47d5487d8ul, 0xd274f4720cd61725ul, 0x711b6f90440eefc5ul, 0xeef2ca609048c4f9ul, 
   0xffba8c979f1ca7e0ul, 0xb348550b2ec10600ul, 0x909c874a9b7c6f71ul, 0x10dc31facb1dc6c4ul, 0x8e88f6f0f1b2b689ul, 0x8a5ccf7df40ccc72ul, 0xc615caa27094805ul, 0x4cf23d1335ea7026ul, 0xd136d7fd110548a3ul, 0xd974b8404e703d80ul, 0xa5b683a14cdefef5ul, 0x740a135c811434ful, 0x3efcb71191c5f868ul, 0x7a8e5f8a88da7c73ul, 0x8bf35342a56451e7ul, 0x83f68e75f8ddb52ful, 
   0x21d00b37d7dbed02ul, 0x3344edcaed70f4a4ul, 0x8b2d87217cd3519ul, 0x95419264c8cd5fbdul, 0x1224496478db83a8ul, 0x55b375f49ac9f8d8ul, 0x81310817212138e7ul, 0x31395823f42c25caul, 0x27606cc048ef6e81ul, 0x6cc2c2ad3300fffcul, 0xc794ec4c436d4c7dul, 0xb707ac322d2130c6ul, 0xbdc78c0896ca6ec6ul, 0xfd248e3149673c8eul, 0x7c54e22c1205d8beul, 0x938f44d926b03563ul, 
   0x1c2a62abbb01c322ul, 0x45fe3b7fb24e0d00ul, 0x18cf93b4208ee6fcul, 0x52a428b35a0dcfaeul, 0xfb890072b724de50ul, 0x9c5fea36003ee17bul, 0xce07c16734a05f9dul, 0xd39cf2615214781dul, 0x9e08f5cbf1a1f13ul, 0x22b938826cefaf39ul, 0xefe0e5448b3fce6ul, 0xc9874823c002eb8ful, 0xdecfd82d90be19e7ul, 0x57269f0c691ca100ul, 0xefa70658002429dful, 0x5a4784ef6f273a22ul, 
   0xff407ec6f5aa72ul, 0x9164f1e34558c677ul, 0x8481b5102019e9a4ul, 0x5fa87228f95e662aul, 0xf81049ae14d81e95ul, 0x199c77f500c50bcaul, 0xac425031dd1d4ddbul, 0xa84d0fd100b1fecbul, 0x52b06abd5506caf4ul, 0x41d292f98f195bdbul, 0xb7b7e86c02c654a7ul, 0x348c6bb20e682b19ul, 0x3551f0bd92f219a7ul, 0x3e5522a885694b82ul, 0xe28538d19b46cb70ul, 0x774f6b83b2505cc7ul, 
   0xcefa145ed7811215ul, 0x670a2ce2300d0105ul, 0xdad78ffc72e77620ul, 0x6925005cbd8db468ul, 0xafa634fe85643aacul, 0x8e6987385a21bea7ul, 0x93f3ef95683e3858ul, 0x88c18a49c4ad52fdul, 0x9cc09bd94d79f354ul, 0x503bd83396bfdd31ul, 0xb8d99882c8fbe523ul, 0xce4672da65fcfe7dul, 0x919408c5bb29a28bul, 0xa1194eb5628c1de3ul, 0x67aa53547cdf6da0ul, 0x6641d3c9ec252405ul, 
   0xefbd93743960d471ul, 0x325faefbf70f6b29ul, 0xc1fdc7e03c8677a1ul, 0x20e7f7f5b07f53c4ul, 0x90042ec43077b111ul, 0xe7f790db01638d4eul, 0x1272567f5c720125ul, 0xa592c3f6985f2b72ul, 0x4ea9c53bb626a28ul, 0x9f7a7e270de34bb1ul, 0xc6169c43db206184ul, 0xad3499064c5d0c20ul, 0x8176b98c08430918ul, 0x124f6386953efb91ul, 0xed649e75aac1f21ul, 0xf73ef40d23339b31ul, 
   0x8cd66d96001a9217ul, 0xc3bb31484834fb63ul, 0x26985068322e3fd0ul, 0xeb185d0e03d3758eul, 0x5b01725ddb1aa60bul, 0x101a44481cc6077dul, 0xcc7f4ccdc490b139ul, 0xfa71ac5787099969ul, 0x7941b222dcd226abul, 0x443895ef59ea713aul, 0xd6b946d788128233ul, 0x6267c8980e0dad23ul, 0x7aaaab00d7e3c24eul, 0x62b4f72086b5c8f6ul, 0xde4f5041343c0b27ul, 0x8c7061cd9a0e9441ul, 
   0x70f7627c3bba2981ul, 0xdbf3d8d891318a33ul, 0x9cf8a9d73c06f05dul, 0x89c8f86d129e2704ul, 0xbc1bbed7f4333c09ul, 0xbe0fc6181feaaa98ul, 0x18c65db6663cdd02ul, 0xf238077d33264cd0ul, 0x68d38b2bc1c2f7b9ul, 0xd1d6278b91c661dcul, 0x5b5b321a003c71aeul, 0xd312169989cc6a49ul, 0x855dcf214c8551aeul, 0xa4bd828e0c09a66bul, 0x8c1f93540e951187ul, 0x4026e789784cc98ful, 
   0x316205eff79f2362ul, 0x11d74d5963458407ul, 0x163d19572f1cdb2ful, 0xc83f1b88d125cd1ul, 0x4ebfe1343aea02dbul, 0xe937038cf71211ceul, 0x78da189ee68cfeceul, 0xcf2e73d87ecaa063ul, 0x633d522d01cc9c2cul, 0x339f8e13c1f36220ul, 0x4ec872c648076895ul, 0xadea51d163bc63d2ul, 0xcb74a725d9930cccul, 0x8912a918571138aful, 0x4412a833dc3d01e9ul, 0xbc9f73668af72947ul, 
   0x9b430267ecd5ac97ul, 0x5b26266b6dcefd0ful, 0x9388170027f4906bul, 0xc0c0c8753c95df43ul, 0xba5798aa82ebf0eful, 0x2102aa28e715adf3ul, 0xa0c580a317528e66ul, 0x39cc7026f629cc5ul, 0xa1abc7f93305ce0bul, 0x1e9f4c0585efe527ul, 0xc55432e905472f3cul, 0xb8818a9b9729398ul, 0xa6411a188e57f24ful, 0xe9e1470413929000ul, 0x94f713d908182983ul, 0x45e2c9e4afe61bb3ul, 
   0x15312818bf85208aul, 0x271e88a5225c0f79ul, 0x8f6219b4c6c47326ul, 0xc57934952ac32293ul, 0x79639ffaa606c449ul, 0x3c69e27c276fe9e9ul, 0x51c24d2b8171b42bul, 0x4c6324c4958f0c2ful, 0x2ccef40af1095988ul, 0xeacc4abe4f1cc129ul, 0x62fcae75e412b04bul, 0xb3ed4c09b966acf8ul, 0xa63c4346e914364eul, 0x4b8c9a274a74e24dul, 0x5dc9f986c4c22130ul, 0xc87a2f9b70732acful, 
   0x4f8e1e996d4d729bul, 0x2be387145ee4deccul, 0x7100ffe9f6e8d2e7ul, 0xf78d98739656041ul, 0xe618cfb570f0a96ul, 0xaa3aae9a00f0658ul, 0x13911939323b2b78ul, 0x31f801f621e17ad2ul, 0x75197fd6a3141955ul, 0x89738c6fe31dd811ul, 0xc888eda3c74ca1a1ul, 0x8cb163a4c9d127e0ul, 0x8e73a4d426d53c10ul, 0xc9fa380a645d93d0ul, 0x8a02c23b42cc0be1ul, 0x5edac0b9373c2aa6ul, 
   0x28b3b8cee2d7cbbdul, 0x458d7750adac016ul, 0x3d2c5f3ea40ac999ul, 0xc6ac8afba0b248c1ul, 0xef03bf6dd953040cul, 0x5bad8b5bc8fc8575ul, 0xb76b2c1254cb323cul, 0xc57563314998387ful, 0x69a20b1c627b54d4ul, 0x9c1477657d4ff21ful, 0x288bb39d7d0f4522ul, 0x142c7078cc494da2ul, 0x388f7200ffb4526cul, 0xb5443d618961eedul, 0x9facecc17d81051ful, 0xcb32b23c1cad35a1ul, 
   0x4ab417f2458d8152ul, 0xffeae40d1b5ba1ul, 0x808e2443570859a2ul, 0x119428640a1ad1c9ul, 0x40078111f1d5ca9dul, 0xc9f9bd713a795e8eul, 0x9e6d4ed8dfcbec35ul, 0x82f1431efadace9bul, 0xd7f72e4c4d00fb41ul, 0x1f6960c796ec1836ul, 0xf7208a93797797aaul, 0xac9aad1cd111d170ul, 0x876924bca6c2e0bful, 0x8c2f9a63972d606eul, 0xab9d2eeac8f4d0a0ul, 0xad2187c2ba0b05d7ul, 
   0x418c41a39e5c1ef1ul, 0x97881791ac339398ul, 0xf479f27d3d3512bbul, 0x1a8c8e9fd965782dul, 0x8670b835111295b6ul, 0x5f3b485a3bf10079ul, 0xd8590068e8994e6eul, 0x9a03332a9e376e91ul, 0x553de89c9ab0b563ul, 0xe78a209f9b15a757ul, 0x9a82963c7e9cf944ul, 0x14667a0b0c382291ul, 0x9d53158c901aa4a9ul, 0xc14c72be357df380ul, 0x78b3c6781d4a8714ul, 0x898c2bd2ec2d2713ul, 
   0xc41c314ffa3981b7ul, 0xeac46a8141cd8ff3ul, 0x66c515cdd560fd77ul, 0x576a5f64143176d8ul, 0x4e28f91a0fab9ca7ul, 0xef5725c463062d1eul, 0x379b3518891099acul, 0xb5916c040f490cbcul, 0x19d8e9930a2b4d86ul, 0xe394389d5b513b15ul, 0xaf1146ef1c30472bul, 0x1a59f179dd0d77c5ul, 0xb3dd7e47c4d3eaf0ul, 0xc5381f111eb45e52ul, 0x6de1a9f4fb249c7ul, 0x2ebf9944bc126d24ul, 
   0x97c2994edf0c39f2ul, 0x18aa5b726e67f857ul, 0x14d980213c596bb0ul, 0x2f0289a427679f18ul, 0xa257d624197f6798ul, 0x31508878a5d842a3ul, 0x591012b9c9dde932ul, 0x7a775c1393e54b6dul, 0x9805dd48c6f8cb37ul, 0x42b52e3366fd35e0ul, 0xe84493d82e9b53cul, 0x74d8002de329f78aul, 0x7fa09ee471161f72ul, 0x87be9c4e33713d33ul, 0x6d5aaae386eb5373ul, 0x68236e262f66aa6ful, 
   0xa449315d7323a22eul, 0x48f8703e93222593ul, 0x1c192cab915ad2bul, 0x9a791920dbd9c96dul, 0xdaff7fc6de0ee9ul, 0xaf10000000010108ul, 0xce35d4547fb38e3eul, 0xd261929448e67502ul, 0xd1d00d71de873d1ul, 0xe63aa0032001c884ul, 0x9d1c1df6e863640ful, 0x102500c4ff1f00fful, 0x200030003000101ul, 0x101010302ul, 0x6151413121001101ul, 0xe1f0d1b1a1918171ul, 
   0x1010800dafff1c1ul, 0x698522c103f0100ul, 0x6b38052608e80ef4ul, 0xce6f789b45c491b4ul, 0x6e862d7425ca45aul, 0xb593052ded3596bful, 0xd995bfb9f060df62ul, 0xc766f085400636b0ul, 0x7b101a2e3aba00fful, 0xe02601540bf8e0fdul, 0x2b45bc322eaaa720ul, 0x14ae6b4811902735ul, 0x767a60d3546f719eul, 0x663068de15ee8d4bul, 0x4f8d49c5653d78d0ul, 0xde52ca0c26083846ul, 
   0x92a0652717fa79ful, 0x97b81318ebdffd48ul, 0x76ac37fa5239c1b8ul, 0xb35fa87e005afcb2ul, 0x58f820d928ed9b78ul, 0x1b481d5a4222b31aul, 0xc19bc7cdef362804ul, 0xc39a2f3d4a48321ul, 0x4f251c521a2c22b9ul, 0x7448158db9e9bb12ul, 0xf0294c7d6b88ca40ul, 0xc37b227fdd81c938ul, 0x49c62a06ba8d6c4cul, 0xbca5bf91738c9ab7ul, 0x7b1ad029589b5454ul, 0x26bd6e8145ebbc8ul, 
   0xf3f82b45de3c7b58ul, 0x8634355814653b85ul, 0xa5a2dd2c5793b8cul, 0x491687f9fa6d56bbul, 0xa7f3ed7a01cf4f83ul, 0x4e7e5f6b4b7a477cul, 0xab8e7a114416d0f9ul, 0x66fe5a918cfeee5ul, 0x4bfc30fa0afaa938ul, 0x84ef5b7d65ed6021ul, 0xd7c5a74eb2e2656bul, 0x87a18a8c439e4e2ul, 0xd68e8c36e36a6e36ul, 0x688dcbf16567701bul, 0x319ca40a4f083069ul, 0x5f839b168e54d447ul, 
   0xee356b414745e35eul, 0x2d1bcd851e5953bul, 0x262823da9377c773ul, 0x8454e9c4fd504180ul, 0x74aa398fd19f0781ul, 0x95452a11edb229a6ul, 0xb9a6e0f3314f44b4ul, 0xaf93b14e71a5a5fdul, 0x285129883dbdc5dul, 0x601e75978376c3dul, 0x784ff207bf06722ful, 0x5eb3c87c84b8b974ul, 0x888d96131b2a58b2ul, 0x1bf990a296c7d5ebul, 0x1c49cfe2dd00ff09ul, 0xe0f08fb150c8e800ul, 
   0xfe6d3a344861809bul, 0x783c52327b4b7030ul, 0x349dbe36e427cff9ul, 0x8b2449f9ca9df1edul, 0x36a611b7e140c36bul, 0x22e0f0e7c0234f25ul, 0xdedd34a83c82bb2cul, 0xa86bbd82e960d2dcul, 0xa0a78d6d19c6d7f8ul, 0x54e26cf591b42b0ul, 0x62f45e361a2d7a09ul, 0xbd559b62439df6ccul, 0x5d00c4288808d6f9ul, 0x4d0061711f7ce9a1ul, 0x520098bdc1de806aul, 0xb770f45bd0456152ul, 
   0x53a8b8392000801bul, 0xe54812868b317b4ful, 0x3aa4b9dec0171b79ul, 0x617f1d7b0a363555ul, 0x5ea700096d5e21cful, 0x2c5807f465c47ab8ul, 0xfdd7bb901f4aa058ul, 0xeb27a877591d6eacul, 0x10bd122fc8442813ul, 0xb7291216567e2d18ul, 0x4aa0ad3aa9b3c0cdul, 0x305d1d03bf2607cul, 0xa01469051613ef7eul, 0xd90319dc36545fb1ul, 0x7155aea495959252ul, 0x762ef914fa838568ul, 
   0xe30b3f997f532e14ul, 0x1fe58f429b720959ul, 0xfa7898d2fa3bcd7dul, 0x8dbcb6eb1252e677ul, 0xaca100cc8ef57ea8ul, 0x7cb01746df52c227ul, 0x9311ffdddb4b98ful, 0x91099d2f50a0b3b1ul, 0xff7c57546274aful, 0xbdf34b77c91b598ful, 0xc3f31b0c150dcdfdul, 0x7e3651379bf585cful, 0xad3e18eb2f7faful, 0x6f3964449735eb3cul, 0x56b94dec58cf5e7ful, 0x12fd64cfcef0e8ecul, 
   0x2c11436f037a91a6ul, 0x7b81116303a482aaul, 0x2dd51bb0764315acul, 0x52d4035aac701bf4ul, 0x1d70361c0f34983cul, 0xd17495f1afafaa20ul, 0x6102692230df540eul, 0xfc1af443a0874d7aul, 0x309e07409db185e7ul, 0x5ce0a76ea1062448ul, 0x2cfb035800a88117ul, 0xaa8a0597ccf3fa3aul, 0x3a20b7f2c6b9f8c3ul, 0x69b4f257a17a772aul, 0x46af5243a55bb925ul, 0x9ecead8a3161629aul, 
   0x7c1d6ae946319f5ful, 0xd526b123f77ff83dul, 0xa2a9472fef7ad2bul, 0x4d0d5800fffab7d7ul, 0xcc00ff787d9e2a6cul, 0x7fd9f36850df917aul, 0x693e9aa437cbad59ul, 0x9b8f3dd8488b8231ul, 0x443c2023dd8dd7c0ul, 0x7222e7115ed3d3d3ul, 0xf1f57e9756355485ul, 0xff1005ddd286c1c2ul, 0xdf5632f20463ea00ul, 0x1524b149c0ec41a3ul, 0x6800d62100ff5d07ul, 0x60621edf3da9c0f4ul, 
   0xfcbf780058b99b65ul, 0x5cd335d045ece970ul, 0xd1e66916f660f6aeul, 0xaed1f507d07089e5ul, 0x3796b4e00ff1ac2ul, 0xc6247632403681a1ul, 0xfdf11d858e46d7aaul, 0xf8d7ac464bd54bbful, 0xde7ffe3555901e32ul, 0x321f3dbf7635383dul, 0xc8c479983d80ce48ul, 0xbceb719ecf464c55ul, 0x1c2a83bbbe9ac827ul, 0xc5dacafad1b0c541ul, 0x729e0d577d331a05ul, 0x2300ffd88496b906ul, 
   0xd6b0a858693dadbcul, 0x63132f43d58739ecul, 0xda62409875b58ebul, 0x13dc0acbf9cea99aul, 0xa4cf9cda146a3dcul, 0x3e00ff6bec52ab05ul, 0xbe8e270852a55b63ul, 0x40d1562b37ebdba0ul, 0xd0d959df1e962daful, 0xfb235eebe42f682aul, 0x2c8373413c6cdcaul, 0x1f689243b833f7acul, 0xa800c955989dbf60ul, 0xd946cbc4dbb6d05eul, 0x2e888224e33f4f1bul, 0xef9e2768a5a96eeeul, 
   0x66e7b73fa0890d2eul, 0x7ffdd57a93b06435ul, 0xf1f9e76d168b1518ul, 0xe534c7944e8d8e8aul, 0xdc9d63faa9a36845ul, 0xe3c48b727d7fba6aul, 0xf492f74f45d04aa0ul, 0x6ec08dcb679f213cul, 0xacd953a102da0cf0ul, 0x50000c1779665737ul, 0x5b48768793c9ad8bul, 0x88a8acf52e77c347ul, 0x321e61442b755110ul, 0xfebfec00805d00e0ul, 0x1d31806d448bf9b4ul, 0x44ed629dbf172c8aul, 
   0x62e21ffeea7b9432ul, 0xf760576ba001af7ul, 0xe26f50484e843da5ul, 0x8ae22460e87a3b40ul, 0x7fa5db5d69449c3ful, 0xcdea887178ea7124ul, 0x110e6c4a2e388645ul, 0x36efcffe31f26213ul, 0x5f17afc40d0c0de9ul, 0x21c7e8de9cce1f1bul, 0xcfe00afb1ed00a84ul, 0x82e9f72db75697f1ul, 0xd6bfbe44b4260a61ul, 0xddd6e5a08d90713ful, 0xb0e824d45088f1eful, 0x4042088defd35a10ul, 
   0x2bd537b590e2bf4dul, 0x5e9601b509a455c2ul, 0x2b900ea3f2436e77ul, 0x26eea44abef574e2ul, 0xd31de2b35a536b44ul, 0x9f80b693586ca8f6ul, 0x131eadfc4a6e8ca7ul, 0x39035780791cae45ul, 0xe701c2140debbfbeul, 0xc33a6c03000de559ul, 0xb190646340019508ul, 0x2ccabb14f48028d2ul, 0x79044feefb471141ul, 0x47695d954de3f00ul, 0xf6b7d19238f6978ful, 0xe6fc7d75d86b31d6ul, 
   0x410774902895ef29ul, 0x47b6232d8772fe78ul, 0xa2343913e3fc2780ul, 0xba1d7095a7b3d29bul, 0xcc672f8de9b20676ul, 0x62c04657172d238cul, 0x689aa91b9831a291ul, 0x6eb5d2b164917baul, 0xc834a250a0fc2128ul, 0x299a20cf116ebd2ul, 0xe100d4c97b3435baul, 0xd3717f730ddd7561ul, 0xbe5bf0d7e38e25c5ul, 0xfd4905745162c7f9ul, 0x3726768b4209377ful, 0x80970ee0aaf507baul, 
   0xb88600ff15f37ff3ul, 0xa6e579e5a30c0daeul, 0xa60f54019fa886e8ul, 0xaa02fcf2d4c59de2ul, 0xa2804d9c60a00ad0ul, 0xfe4f8e8f765bc687ul, 0x4cb9fd81be4cce44ul, 0xb77b1a29b148f595ul, 0x16b86b8d5a0b8c23ul, 0xfab74fc11349faf2ul, 0xf6f2176063a470c9ul, 0xcd4213630dce5fbaul, 0x7e112334af9cf0b4ul, 0xee38a01b97df7313ul, 0x78ce6deafa40f6d4ul, 0xe1fa1eab7342240dul, 
   0x3c6aee0d3790ca95ul, 0xd2ac2a6cb6b995d4ul, 0x75df4b301f24287bul, 0x688c96b91f7a3c06ul, 0x3bd977015e64472aul, 0x7e3b5d5d9723309aul, 0x8e7ea60d8bc02c97ul, 0x73c301b17ec4fabeul, 0x12e15e62ed00ffdeul, 0x74380e68523b162aul, 0xd408a6540764a242ul, 0xfa001e043a17ac55ul, 0xed06a8806e63e245ul, 0x900d906e8509754ful, 0x3b38d6114ffbb088ul, 0xb8327aeeeb0f4171ul, 
   0x569496f99b0c1528ul, 0xe63dece605e45191ul, 0xf61087de564af185ul, 0xaba1017120c7344eul, 0x9d9a5263da3e9a66ul, 0x2ee06a340c8a1193ul, 0x1b87532137f6b44aul, 0xd0cd2d772b1dc8a8ul, 0x30037a4f9560d85bul, 0xbf303feb0ee4e9d0ul, 0xb47ddb14403287ful, 0xb96be741218b9269ul, 0xc16e47639085e37cul, 0xe28fd24b34886aaaul, 0x7bc5e4bbbbc3823dul, 0x557d2bad5108918ul, 
   0xa26f21509891384ful, 0xd5c0604e27a503c7ul, 0x9d9db5760c476903ul, 0xb82ac3c82a4a0c6ul, 0x3c859a488b260146ul, 0x69d60977771e28ebul, 0x9aaab4c9caf93bf2ul, 0x84bd986b8e4f4f43ul, 0xc2f5f9601b80a543ul, 0x83eb9fee73b02e5ul, 0x384257db60a9941cul, 0xffa6210340599563ul, 0x97836002f9965e00ul, 0xa09d66b3992d8debul, 0x467b805c37a290aul, 0x601db323e335d48aul, 
   0x1bbcb2b45aeb6947ul, 0x254686582132821ul, 0x10abc40ed4bce318ul, 0xa8c98f564b93bc48ul, 0x92500e8b8a94190ul, 0x1e415f137b141ful, 0x43170813d0a10351ul, 0x546df29c9e4515c0ul, 0xe0e96d094a9a21a4ul, 0x42a9621301f91ab3ul, 0x1060b6ec0a7034e9ul, 0x90afc85f120557cul, 0x570ce14476b09864ul, 0x7756ad0a888a2100ul, 0x7ab2c83e507d6218ul, 0x89bbe98c8f799fa2ul, 
   0xf9e6749b52f90015ul, 0xb69a5d84d8420962ul, 0x7b98aabbc406355cul, 0xd0675ddb3bc9fb4bul, 0x37fa429a3bac40d9ul, 0x93c64314341adc6bul, 0x20a9cae6c4f8da03ul, 0x28111b31202f6ed9ul, 0xdc9afcb8da95750bul, 0x23519385b60c8978ul, 0x4eda2891333a2426ul, 0x28104b00da73390ul, 0x6897cdd9a48fe850ul, 0xe1231c04b84f147ul, 0xb5b1099fd8a24d10ul, 0x2898e0bbea6a6ea6ul, 
   0x1a1118d3ccdb011aul, 0x269cee1645179bd0ul, 0xa050ba225125358eul, 0x3a49476b5ea375bbul, 0x1fa88f33b8ade6cul, 0xb8ea533d42291808ul, 0x6d29000418c96efeul, 0x8c2d368098ddd1ul, 0x2afc246b4238c02ul, 0x90a20f1a294c05d5ul, 0x4840f96847b4a670ul, 0x5b40f1d13c81c579ul, 0x84c46b03a28d2288ul, 0x4ac392fd4e24f540ul, 0xccc2478cf06a92a7ul, 0x6614045a323dd48aul, 
   0x337a18964704468dul, 0xbc7aaf35930e41a1ul, 0x9a0190bc53830a2dul, 0xdf38e05fc3bba1e5ul, 0xca4628960dc26891ul, 0x10a01b0a69585f88ul, 0x89f1c7cd120f8e79ul, 0x862052a4da0a086dul, 0x6b0a001b66b96460ul, 0x260850f0b0b9bcbul, 0x8f23ee2e26fdc85ul, 0xf205a0ee91dfb201ul, 0xa185068126d2dc81ul, 0xb240aa1007947504ul, 0x6b1c807a00579200ul, 0x9da65c0d83211de5ul, 
   0x684402ab85daa522ul, 0x1c2ce29ada53674eul, 0xaabc4f4034237a32ul, 0x516a655dd44b41e9ul, 0xbbe9030d511ddc56ul, 0x6f5a60ab0e0e9815ul, 0x804867f58de17c67ul, 0xb1438ac1b78b86a4ul, 0x460ca7284bda1b8aul, 0x87ee103450bd41b5ul, 0xa0c03a01772bc82cul, 0xdb12c928044f9e00ul, 0xd58c9abd89360115ul, 0x76b50ad35243229eul, 0x15667c6f6173d08aul, 0xb138aceb3734a01ful, 
   0xcabdd1b6b6d1bb35ul, 0xc483101a8bfda259ul, 0x423ac00d75c7447eul, 0x14b4d38ef15136a1ul, 0xe489d9ecd2111403ul, 0xa553a8532a10e53aul, 0x63819a4024855e88ul, 0x6c03faa64c3c1665ul, 0x61eee0baaff2491bul, 0x27d57dba473441e6ul, 0xa4c75630220cca2aul, 0xac4082b45e87e5d4ul, 0xaa893ed0a5796693ul, 0x726944ad8f8a256cul, 0x31a85def34026050ul, 0xf2b6200608a4bb4dul, 
   0x9fea29e10841d01bul, 0x7c6b0d3863550a09ul, 0x8b54a594a6a7602dul, 0x41800840169c8919ul, 0xe811e1481607fee1ul, 0x5d0713e90ec356d1ul, 0x6dc036ca06f6d8a0ul, 0x2d82b474dc9b96c2ul, 0x222a5d0b27d7b285ul, 0xccddc2d7098cb49dul, 0xc5f14277c3e541d0ul, 0x3a885656826a0134ul, 0xaa893ed0d9ffaca7ul, 

};
const std::size_t data_inputImage_size = 10084;
