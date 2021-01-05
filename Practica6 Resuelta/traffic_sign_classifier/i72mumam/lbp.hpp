//! \file lbp.h
//! Utils for LBP

#ifndef _FSIV_LBP_H_
#define _FSIV_LBP_H_

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/ml/ml.hpp>
#include <iostream>
#include <vector>

using namespace std;
using namespace cv::ml;

/**
 * @brief Computes a LBP matrix
 * @param[in] img Input image (gray scale)
 * @param[out] lbp image.
 * @param[in] uLBP if true uLBP is computed.
 * @warning: The most significative bit is for the comparison p(x-1,y-1)>p(x,y).
 * @pre !img.empty() && img.type()==CV_32FC1
 * @post lbp.type()==CV_8UC1
 * @post lbp.rows==img.rows && lbp.cols==img.cols
 * @post !uLBP || ((min_code>=0.0) && (max_code<59))
 * @post uLBP || ((min_code>=0.0) && (max_code<256))
 */
void fsiv_lbp(const cv::Mat & img, cv::Mat & lbp, const bool uLBP=false);

/**
 * @brief Compute a lbp histogram.
 * @param[in] lbp input image.
 * @param[out] lbp_hist as a row vector.
 * @param[in] normalize true for normalize the histogram.
 * @param[in] uLBP true if the input lbp image has uLBP.
 * @pre !lbp.empty() && lbp.type()==CV_8UC1
 * @post lbp_hist.type()==CV_32FC1
 * @post lbp_hist.rows==1 && (uLBP ? (lbp_hist.cols==59) : (lbp_hist.cols==256))
 * @post !normalize || (std::abs(cv::sum(lbp_hist)[0]-1.0)<1.0e6)
 */
void fsiv_lbp_hist(const cv::Mat & lbp, cv::Mat & lbp_hist,
                   const bool normalize=true, const bool uLBP=false);

/**
 * @brief Computes a LBP descriptor (concatenation of cells)
 * @param[in] lbp input image.
 * @param[out] lbp_desc as a row vector.
 * @param[in] ncells array with [row x cols] e.g. {2,2}
 * @param[in] normalize true for normalize the cell's histograms.
 * @param[in] uLBP true if the input lbp image has uLBP.
 * @pre !lbp.empty() && lbp.type()==CV_32FC1
 * @pre ncells!=nullptr && (ncells[0]*ncells[1] > 0)
 * @post !lbp_desc.empty();
 * @post lbp_desc.type()==CV_8UC1
 * @post lbp_desc.rows==1
 * @post uLBP  || lbp_desc.cols==(256*ncells[0]*ncells[1])
 * @post !uLBP || lbp_desc.cols==(59*ncells[0]*ncells[1])
 * @post !normalize || abs(sum(lbp_desc)[0]-1.0)<1.0e-6
 */
void fsiv_lbp_desc(const cv::Mat & lbp, cv::Mat & lbp_desc,
                   const int *ncells, bool normalize=true, bool uLBP=false);


#endif
