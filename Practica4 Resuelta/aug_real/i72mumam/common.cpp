#include "common.hpp"
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <iostream>


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
bool fsiv_load_intrinsic_data(const std::string& filename, float & error, int & height, int & width, cv::Mat & M, cv::Mat & dist_coeffs)
{
    bool ret_val = true;

    cv::FileStorage fs(filename, cv::FileStorage::READ);

    

    fs["image-width"] >> width;
    fs["image-height"] >> height;
    fs["error"] >> error;
    fs["camera-matrix"] >> M;
    fs["distortion-coefficients"] >> dist_coeffs;

    if(width == 0 || height == 0 || error == 0)
    {
      ret_val = false;
    }

    //Hint: User cv::FileStorage class to manage this.

    return ret_val;
}

/**
* @brief Write the intrinsic data to a file.
* use 'error' for calibration error, 'image-height' for image's height,
* 'image-width' for image's width, 'camera-matrix' for
* the camera's matrix and 'distortion-coefficients' for
* the distortion coefficients.
* @arg[in] filename of the file where the data is loaded.
* @arg[in] error is the reprojection error returned by calibration.
* @arg[in] height is the image's height.
* @arg[in] width is the image's width.
* @arg[in] M is the camera matrix.
* @arg[in] dist_coeffs are the distortion coefficients.
* @return True if success.
*/
bool fsiv_write_intrinsic_data(const std::string& filename, const float error, const int height, const int width, const cv::Mat M, const cv::Mat dist_coeffs)
{
    bool ret_val = true;

    cv::FileStorage fs(filename, cv::FileStorage::WRITE);

    fs << "image-width" << width;
    fs << "image-height" << height;
    fs << "error" << error;
    fs << "camera-matrix" << M;
    fs << "distortion-coefficients" << dist_coeffs; 
    //TODO
    //Hint: User cv::FileStorage class to manage this.

    return ret_val;
}

/**
* @brief Compute the 3D world coordinates of the inner board corners.
* @warning The first corner will have coordinates (size, size, 0.0).
* @arg[in] rows are the number of board's rows.
* @arg[in] cols are the number of board's columns.
* @arg[in] size is the side's length of a board square.
* @return a vector of with (rows-1)*(cols-1) 3d points following a row,
cols travelling.
*/
std::vector<cv::Point3f> fsiv_compute_object_points( const int rows, const int cols, const float size)
{
    std::vector<cv::Point3f> obj_points;

    for(int i = 1; i < rows; i++)
    {
      for(int j = 1; j < cols; j++)
      {
        obj_points.push_back(cv::Point3f(size * j, size * i, 0));
      }
    }

    CV_Assert(obj_points.size() == (rows-1) * (cols - 1) );
     
    return obj_points;
}

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

void fsiv_draw_axes(const cv::Mat& M, const cv::Mat& dist_coeffs, const cv::Mat& rvec, const cv::Mat& tvec, const float size, cv::Mat img)
{
    CV_Assert(img.type()==CV_8UC3);

    std::vector<cv::Point3d> reprojectsrc;
    
    reprojectsrc.push_back(cv::Point3d(0, 0, 0));
    reprojectsrc.push_back(cv::Point3d(size, 0.0, 0.0));
    reprojectsrc.push_back(cv::Point3d(0.0, size, 0.0));
    reprojectsrc.push_back(cv::Point3d(0.0, 0.0, - size));

    // puntos 3D proyectados en 2D
    std::vector<cv::Point2d> reprojectdst;

    // reproject
    cv::projectPoints(reprojectsrc, rvec, tvec, M, dist_coeffs, reprojectdst);

    // dibujar los ejes de rotacion 
    line(img, reprojectdst[0], reprojectdst[1], cv::Scalar(255, 0, 0) , 2, cv::LINE_AA);
    line(img, reprojectdst[0], reprojectdst[2], cv::Scalar(0, 255, 0), 2, cv::LINE_AA);
    line(img, reprojectdst[0], reprojectdst[3], cv::Scalar(0, 0, 255), 2, cv::LINE_AA);
}

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
void fsiv_draw_3d_model(const cv::Mat& M, const cv::Mat& dist_coeffs, const cv::Mat& rvec, const cv::Mat& tvec, const float size, cv::Mat img)
{
    CV_Assert(img.type()==CV_8UC3);

    std::vector<cv::Point3d> reprojectsrc;

    reprojectsrc.push_back(cv::Point3d(0, 0, 0));
    reprojectsrc.push_back(cv::Point3d(size, 0.0, 0.0));
    reprojectsrc.push_back(cv::Point3d(0.0, size, 0.0));
    reprojectsrc.push_back(cv::Point3d(size/2,size/2, - size));
    reprojectsrc.push_back(cv::Point3d(size, size, 0.0));

   // cv::Mat rmat;

    // puntos 3D proyectados en 2D
    std::vector<cv::Point2d> reprojectdst;

    // reproject
    cv::projectPoints(reprojectsrc, rvec, tvec, M, dist_coeffs, reprojectdst);

    // dibujar los ejes de rotacion 
    line(img, reprojectdst[0], reprojectdst[1], cv::Scalar(255, 0, 0), 2, cv::LINE_AA);
    line(img, reprojectdst[0], reprojectdst[2], cv::Scalar(255, 0, 0), 2, cv::LINE_AA);
    line(img, reprojectdst[0], reprojectdst[3], cv::Scalar(255, 0, 0), 2, cv::LINE_AA);
    line(img, reprojectdst[1], reprojectdst[4], cv::Scalar(255, 0, 0), 2, cv::LINE_AA);
    line(img, reprojectdst[2], reprojectdst[4], cv::Scalar(255, 0, 0), 2, cv::LINE_AA);
    line(img, reprojectdst[0], reprojectdst[3], cv::Scalar(255, 0, 0), 2, cv::LINE_AA);
    line(img, reprojectdst[1], reprojectdst[3], cv::Scalar(255, 0, 0), 2, cv::LINE_AA);
    line(img, reprojectdst[2], reprojectdst[3], cv::Scalar(255, 0, 0), 2, cv::LINE_AA);
    line(img, reprojectdst[4], reprojectdst[3], cv::Scalar(255, 0, 0), 2, cv::LINE_AA);

}

/**
* @brief Project input image on the output.
* @arg[in] input is the image to be projected.
* @arg[in] image_points define the projected board corners on the
output image where the input image must be projected.
* @arg[in|out] is the output image.
* @pre input.type()==CV_8UC3.
* @pre output.type()==CV_8UC3.
*/
void fsiv_project_image(const cv::Mat& input, const std::vector<cv::Point2f>& image_points, cv::Mat& output)
{
    std::vector<cv::Point2f> input_corners (4);
    std::vector<cv::Point> image_points_i (4);

    //TODO
    //Hint: output is a combination of the input image masked with
    //the warpped image.

    input_corners[0] = cv::Point2f(0,0);
    input_corners[1] = cv::Point2f(0,input.rows);
    input_corners[2] = cv::Point2f(input.cols,0);
    input_corners[3] = cv::Point2f(input.cols,input.rows);

    cv::Mat imagen = getPerspectiveTransform(input_corners, image_points);
    cv::Mat aux = output.clone();

    warpPerspective(input,aux,imagen,output.size());

    std::vector<std::vector<cv::Point> > contours;
    
    cv::Mat imageWarpedCloned = aux.clone();
    cv::cvtColor(imageWarpedCloned, imageWarpedCloned, cv::COLOR_BGR2GRAY);
    
    cv::findContours (imageWarpedCloned, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
    cv::Mat mask = cv::Mat::zeros(output.size(), CV_8U);
    cv::drawContours(mask, contours, 0, cv::Scalar(255), -1);
    aux.copyTo(output, mask);

    
}
