#ifndef FSIV__COMMON_HPP__
#define FSIV__COMMON_HPP__

#include <string>
#include <vector>
#include <opencv2/core.hpp>

/**
 * @brief Load the intrinsic data to a file.
 * use 'error' for calibration error, 'image-height' for image's height,
 * 'image-width' for image's width, 'camera-matrix' for
 * the camera's matrix and 'distortion-coefficients' for
 * the distortion coefficients.
 * @arg[in] filename of the file where the data is loaded.
 * @arg[out] error is the reprojection error returned by calibration.
 * @arg[out] height is the image's height.
 * @arg[out] width is the image's width.
 * @arg[out] M is the camera matrix.
 * @arg[out] dist_coeffs are the distortion coefficients.
 * @return True if success.
 */
bool fsiv_load_intrinsic_data(const std::string& filename,
                                    float & error,
                                    int & height,
                                    int & width,
                                    cv::Mat & M,
                                    cv::Mat & dist_coeffs);

/**
 * @brief Write the intrinsic data to a file.
 * Use 'error' for calibration error, 'image-height' for image's height,
 * 'image-width' for image's width, 'camera-matrix' for
 * the camera's matrix and 'distortion-coefficients' for
 * the distortion coefficients.
 * @arg[in] filename of the file where the data is written.
 * @arg[in] error is the reprojection error returned by calibration.
 * @arg[in] height is the image's height.
 * @arg[in] width is the image's width.
 * @arg[in] M is the camera matrix.
 * @arg[in] dist_coeffs are the distortion coefficients.
 * @return True if success.
 */
bool fsiv_write_intrinsic_data(const std::string& filename,
                               const float error,
                               const int height,
                               const int width,
                               const cv::Mat M,
                               const cv::Mat dist_coeffs);

/**
 * @brief Compute the 3D world coordinates of the inner board corners.
 * @warning The first corner will have coordinates (size, size, 0.0).
 * @arg[in] rows are the number of board's rows.
 * @arg[in] cols are the number of board's columns.
 * @arg[in] size is the side's length of a board square.
 * @return a vector of with (rows-1)*(cols-1) 3d points following a row, cols travelling.
 */
std::vector<cv::Point3f> fsiv_compute_object_points( const int rows,
                                                const int cols,
                                                const float size);

/**
 * @brief Draw the world's coordinate axes.
 * @warning the use of cv::drawAxes() is not allowed.
 * Use color blue for axe X, green for axe Y and red for axe Z.
 * @arg[in] M is the camera matrix.
 * @arg[in] dist_coeffs are the distortion coefficients.
 * @arg[in] rvec is the rotation vector.
 * @arg[in] tvec is the translation vector.
 * @arg[in] size is the length of the axis.
 * @arg[in|out] img is the image where the axes are drawn.
 * @pre img.type()=CV_8UC3.
 */
void fsiv_draw_axes(const cv::Mat& M, const cv::Mat& dist_coeffs,
               const cv::Mat& rvec, const cv::Mat& tvec,
               const float size,
               cv::Mat img);


/**
 * @brief Draw a 3D model using the world's coordinate system.
 * @arg[in] M is the camera matrix.
 * @arg[in] dist_coeffs are the distortion coefficients.
 * @arg[in] rvec is the rotation vector.
* @arg[in] tvec is the translation vector.
 * @arg[in] size is the length of the axis.
 * @arg[in|out] img is the image where the axes are drawn.
 * @pre img.type()=CV_8UC3.
 */
void fsiv_draw_3d_model(const cv::Mat& M, const cv::Mat& dist_coeffs,
               const cv::Mat& rvec, const cv::Mat& tvec,
               const float size,
               cv::Mat img);

/**
 * @brief Project input image on the output.
 * @arg[in] input is the image to be projected.
 * @arg[in] image_points define the projected board corners on the output image where the input image must be projected.
 * @arg[in|out] is the output image.
 * @pre input.type()==CV_8UC3.
 * @pre output.type()==CV_8UC3.
 */
void fsiv_project_image(const cv::Mat& input, const std::vector<cv::Point2f>& image_points, cv::Mat& output);

#endif //FSIV__COMMON_HPP__
