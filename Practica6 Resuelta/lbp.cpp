#include "common_code.hpp"
#include "lbp.hpp"


using namespace std;
using namespace cv::ml;


static const uchar uniform[256] =
{
0,1,2,3,4,58,5,6,7,58,58,58,8,58,9,10,11,58,58,58,58,58,58,58,12,58,58,58,13,58,
14,15,16,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,17,58,58,58,58,58,58,58,18,
58,58,58,19,58,20,21,22,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,
58,58,58,58,58,58,58,58,58,58,58,58,23,58,58,58,58,58,58,58,58,58,58,58,58,58,
58,58,24,58,58,58,58,58,58,58,25,58,58,58,26,58,27,28,29,30,58,31,58,58,58,32,58,
58,58,58,58,58,58,33,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,34,58,58,58,58,
58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,
58,35,36,37,58,38,58,58,58,39,58,58,58,58,58,58,58,40,58,58,58,58,58,58,58,58,58,
58,58,58,58,58,58,41,42,43,58,44,58,58,58,45,58,58,58,58,58,58,58,46,47,48,58,49,
58,58,58,50,51,52,58,53,54,55,56,57
}; /**< Look-up table to convert 256 lbp codes to 59 u-lbp codes. **/

void fsiv_lbp(const cv::Mat & img, cv::Mat & lbp, const bool uLBP)
{    
    // assert(!img.empty() && img.type()==CV_32FC1);
    lbp = cv::Mat::zeros(img.rows, img.cols, CV_8UC1);

    //TODO: compute the lbp descriptor.
    //Hint: first compute the 256 lbp code, then if the uLBP is required,
    //use the uniform lookup table to get the correspoing uLBP code.
    //Remember:
    // Threshold using '>' operator, p.e., p(x-1, y-1)>p(x, y)
    // The weights are
    // 2⁷ 2⁶ 2⁵
    // 2⁰ x  2⁴
    // 2¹ 2² 2³
    //
    // The codes for the points that not can be computed will be zeros.

    for(int i = 1; i < (img.rows - 1); i++)
    {
        for(int j = 1; j < (img.cols - 1); j++)
        {
            int x = i; 
            int y = j; 

            uchar value = img.at<uchar>(x,y); 
            uchar aux; 
            uchar result = 0; 

            aux = img.at<uchar>(i-1,j);
            if(aux > value){
                result = result | (1 << 7);
            }
            aux = img.at<uchar>(i-1,j-1);
            if(aux > value){
                result = result | (1 << 6);
            }

            aux = img.at<uchar>(i,j-1);
            if(aux > value){
                result = result | (1 << 5);
            }

            aux = img.at<uchar>(i+1,j-1);
            if(aux > value){
                result = result | (1 << 4);
            }

            aux = img.at<uchar>(i+1,j);
            if(aux > value){
                result = result | (1 << 3);
            }

            aux = img.at<uchar>(i+1,j+1);
            if(aux > value){
                result = result | (1 << 2);
            }

            aux = img.at<uchar>(i,j+1);
            if(aux > value){
                result = result | (1 << 1);
            }
            aux = img.at<uchar>(i-1,j+1);
            if(aux > value){
                result = result | (1 << 0);
            }

            lbp.at<uchar>(i,j)=result;
        }
    }

    assert(lbp.type()==CV_8UC1);
    assert(lbp.rows==img.rows && lbp.cols==img.cols);
#ifndef NDEBUG
    double max_code, min_code;
    cv::minMaxLoc(lbp, &min_code, &max_code, nullptr, nullptr);
    assert(!uLBP || ((min_code>=0.0) && (max_code<59)));
    assert(uLBP  || ((min_code>=0.0) && (max_code<256)));
#endif
}

void
fsiv_lbp_hist(const cv::Mat &lbp, cv::Mat & lbp_hist, const bool normalize, const bool uLBP)
{	    
    assert(!lbp.empty() && lbp.type()==CV_8UC1);

    // Establishing the number of bins
    int histSize = uLBP ? 59 : 256;

    //TODO.
    //Hint: use "cv::calcHist" if you want, but remember we want row vectors so
    //the histogram must be reshaped. You can reshape a cv::Mat using the
    //cv::Mat::reshape() method.
    //
    //Also remember to normalize the histogram if it is required (sum(h)==1.0).

    float range[] = {0,(float)histSize};
    const float* histRange = {range};
    cv::Mat aux;

    calcHist(&lbp, 1, 0, cv::Mat(), aux, 1, &histSize, &histRange);

    if(normalize == true)
    {
        // float num = lbp.rows * lbp.cols;
        lbp_hist.convertTo(lbp_hist, CV_32FC1);
           
         cv::normalize(aux,aux,1.0,aux.cols,cv::NORM_L1,-1,cv::Mat());   
            // for(int i = 0; i < histSize; i++)
            // {
            //     lbp_hist.at<float>(i) = (lbp_hist.at<float>(i) / num);
            // }
    }

    cv::transpose(aux,lbp_hist);
    //
    assert(lbp_hist.type()==CV_32FC1);
    assert(lbp_hist.rows==1 && (uLBP ? (lbp_hist.cols==59) : (lbp_hist.cols==256)));
    assert(!normalize || (std::abs(cv::sum(lbp_hist)[0]-1.0)<1.0e6));
}

void fsiv_lbp_desc(const cv::Mat & lbp, cv::Mat & lbp_desc, const int *ncells, bool normalize, bool uLBP)
{
    assert(!lbp.empty() && lbp.type()==CV_8UC1);
    assert(ncells!=nullptr && (ncells[0]*ncells[1] > 0));

    const int cell_h = cvFloor(double(lbp.rows) / ncells[0]);
    const int cell_w = cvFloor(double(lbp.cols) / ncells[1]);
    const int hist_size = uLBP ? 59 : 256;


    // lbp_desc = cv::Mat(ncells[0]*ncells[1], hist_size, CV_32FC1);

    cv::Mat aux;
    cv::Mat trozo;
    cv::Mat trozolbp;
    cv::Mat histograms;
    cv::Mat trozohist;

    int contador = 0;
    int Theight = cell_h;
    int Twidth = cell_w;
        
        for (int i = 0; i <= lbp.rows - cell_h; i = i + cell_h)
        {   
            for (int j = 0; j <lbp.cols - cell_w; j = j + cell_w)
            {

                contador++;

                    if(lbp.rows - i < 2 * cell_h && lbp.rows - i < cell_h)
                    {
                        Theight = cell_h + (lbp.rows - i - cell_h);
                    }

                    if(lbp.rows - j < 2 * cell_w && lbp.cols - j < cell_w)
                    {
                        Twidth = cell_w + (lbp.cols - j - cell_w);
                    }

                cv::Rect aux(j, i, Twidth, Theight);                          
                trozo = lbp(aux);

                fsiv_lbp(trozo, trozolbp);
                

                fsiv_lbp_hist(trozolbp, trozohist, normalize);

                    if(contador == 1){
                        lbp_desc=trozohist.clone();
                        
                    }
                    else{
                    cv::hconcat(trozohist,lbp_desc);
                    }

                // histograms.push_back(trozohist);
                // contador++;
            }
        }

        // cv::hconcat(histograms, lbp_desc);

    // lbp_desc.reshape(0, 1);

    //TODO: Use fsiv_lbp_hist() for each cell and concatenate the histograms
    // to get the image descriptor.
    //Hint: you can use cv::hconcat() or you can create an initial descriptor
    //with enough space and use cv::Mat::copyTo on it using a roi.

    //
    assert(!lbp_desc.empty());
    assert(lbp_desc.type()==CV_32FC1);
    assert(lbp_desc.rows==1);
    assert(uLBP || lbp_desc.cols==(256*ncells[0]*ncells[1]));
    assert(!uLBP || lbp_desc.cols==(59*ncells[0]*ncells[1]));
    assert(!normalize || std::abs(cv::sum(lbp_desc)[0]-1.0)<1.0e-6);
}

