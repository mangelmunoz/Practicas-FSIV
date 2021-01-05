#ifndef __COMMON_CODE_HPP__
#define __COMMON_CODE_HPP__

#include <string>
#include <vector>
#include <opencv2/core.hpp>

#ifdef NDEBUG
#define DEBUG(l, x) while(0){};
#else
#include <iostream>
#include <iomanip>
extern const std::string __Debug_Padding;
extern size_t __Debug_PPL; //Pad per level.
extern int __Debug_Level;

#define DEBUG(l, x) {if (l <= __Debug_Level) std::cerr << '[' << std::setw(3) << l << "] " << __Debug_Padding.substr(0, __Debug_PPL*size_t(std::max(0, (l-1)))) << x;}
#endif

/**
 * @brief Decode a string like "RxC" to set a grid shape.
 * @param[in] str is the string coding the grid shape.
 * @param[out] n_cells a vector with rows and cols.
 * @return True if success.
 */
bool string_to_ncells(const std::string& str, int n_cells[]);

/**
 * @brief Load the categories to train.
 * @param[in] dataset_pathname locates the dataset in the file system.
 * @param[out] cats is a vector with the class labels.
 * @param[out] descs is a vector with the class descriptions.
 * @return true if success.
 */
bool load_gtsrb_categories(const std::string& dataset_pathname,
                           std::vector<int>& cats,
                           std::vector<std::string>& descs);

/**
 * @brief Load metadata from a GTSRB train dataset metadata file.
 * @param[in] metadata_file is the file with the metadata.
 * @param lfiles will have the training image filenames.
 * @param rois will have the roi for each training image.
 * @param labels will have the class label for each training image.
 * @return true if success.
 */
bool load_gtsrb_train_metadata(const std::string& metadata_file,
                              std::vector<std::string> & lfiles,
                              std::vector<cv::Rect>& rois,
                              cv::Mat & labels);
/**
 * @brief Load metadata from a GTSRB test dataset metadata file.
 * @param[in] dataset_path is the pathname for the whole dataset.
 * @param lfiles will have the testing image filenames.
 * @param rois will have the roi for each testing image.
 * @param labels will have the class label for each testing image.
 * @return true if success.
 */
bool
load_gtsrb_test_metadata(const std::string& dataset_path,
                          std::vector<std::string> & lfiles,
                          std::vector<cv::Rect>& rois,
                          cv::Mat & labels);
/**
 * @brief Normalize an image so mean=0.0 and var=1.0
 * @param src is the input image.
 * @return the normalized versión of src.
 * @pre src.channels()==1
 * @post ret_val.type()==CV_32FC1;
 * @post ret_val's mean is equal to 0.0 +- 1.0e-6
 * @post ret_val's var is equal to 1.0 +- 1.0e-6
 */
cv::Mat fsiv_normalize_mean_var(cv::Mat const& src);

/**
 * @brief Normalize an image so min=0.0 and max=1.0
 * @param src is the input image.
 * @return the normalized versión of src.
 * @pre src.channels()==1
 * @post ret_val.type()==CV_32FC1;
 * @post ret_val's mininum val is equal to 0.0 +- 1.0-e6
 * @post ret_val's maximum val is equal to 1.0 +- 1.0-e6
 */
cv::Mat fsiv_normalize_minmax(cv::Mat const& src);

/**
 * @brief Compute the lbp image descriptors for a set of images.
 * @param[in] lfiles are the filenames of the images.
 * @param[in] rois are the rois where the signs are.
 * @param[in] canonical_size is size to resize the input roi.
 * @param[out] lbp_descs the output lbp descriptors, one row per image.
 * @param[in] ncells sais how to compute compute the lbp descriptor.
 * @param[in] img_norm sais how normalize the input image (0: not normalize, 1:minmax, 2:mean_stdev).
 * @param[in] hist_norm if its true, the lbp histograms are normalized.
 * @param[in] uLBP if its true, the uLBP descriptor is used.
 * @return true if success.
 */
bool compute_lbp_from_list(const std::vector<std::string> & lfiles,
                           const std::vector<cv::Rect>& rois,
                           const cv::Size &canonical_size,
                           cv::Mat & lbp_descs,
                           const int * ncells,
                           const int img_norm=0,
                           const bool hist_norm=true,
                           const bool uLBP=false);


/**
 * @brief Draw an histogram.
 * @param[in] h is the histogram to be drawn
 * @param[in,out] img is the image where draw the histogram.
 * @param color is color of the histogram's bars.
 * @pre !h.empty() && h.type()==CV_32FC1
 * @pre !img.empty() && img.type()==CV_8UC3
 */
void draw_histogram(cv::Mat const& h,
                    cv::Mat& img,
                    cv::Scalar color=cv::Scalar(0,255,0));
/**
 * @brief Draw an image descriptor by concatenating several histograms.
 * @param[in] desc the descriptor.
 * @param[in,out] img the image to draw the descriptor.
 * @param[in] ncells descriptor's grid [0]Rows x [1]Cols.
 * @param[in] color to draw histogram's bars.
 * @pre !h.empty() && h.type()==CV_32FC1
 * @pre desc.rows==1
 * @pre ncells[0]*ncells[1]>0
 * @pre (desc.cols%(ncells[0]*ncells[1])) == 0
 * @pre img.empty() || (img.type()==CV_8UC3 && img.cols > desc.cols/(ncells[0]*ncells[1]))
 */
void
draw_histogram_descriptor(cv::Mat const& desc,
             cv::Mat & img,
             int ncells[],
             cv::Scalar color=cv::Scalar(0,255,0));


/**
 * @brief Compute the size of a file.
 * @param fname is the pathname for the file.
 * @param units is a divisor for get file units. Eg. 1 for bytes, 1024 for Kb, ...
 * @return the file size using the units or -1 if it could not open the file.
 * @pre units > 0
 */
float compute_file_size(const std::string &fname, const long units=1);


#endif //__COMMON_CODE_HPP__
