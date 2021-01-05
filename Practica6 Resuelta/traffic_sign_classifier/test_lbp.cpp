//! \file test_lbp.cpp
//! \author FSIV-UCO

#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>

#include "common_code.hpp"
#include "lbp.hpp"
#include "metrics.hpp"

#ifndef NDEBUG
int __Debug_Level = 0;
#endif

using namespace std;

const cv::String keys =
    "{help h usage ? |      | print this message.}"
    "{uLBP           |      | use uLBP.}"
    "{ncells         |1x1   | Build descriptor using a grid with RxC.}"
    "{@image         |<none>| path to input image.}"
    "{@image2        |      | path to second image, for comparison.}"
#ifndef NDEBUG
    "{verbose        |0     | Set the verbose level.}"
#endif
    ;

int main(int argc, char * argv[])
{
   cv::CommandLineParser parser(argc, argv, keys);
   parser.about("Test LBP implementation.");
   if (parser.has("help"))
   {
      parser.printMessage();
      return 0;
   }
#ifndef NDEBUG
      __Debug_Level = parser.get<int>("verbose");
#endif
   // const bool uLBP = parser.has("uLBP");
   const bool uLBP = false;
   int ncells[2];
   if (!string_to_ncells(parser.get<std::string>("ncells"), ncells))
   {
       std::cerr << "Error: could not decode a valid n_cells CLI parameter." << std::endl;
       return EXIT_FAILURE;
   }

   std::string img1_name = parser.get<cv::String>("@image");
   std::string img2_name = parser.get<cv::String>("@image2");
   if (!parser.check())
   {
       parser.printErrors();
       return 0;
   }
   /// Load the image   
   cv::Mat image1 = cv::imread(img1_name, cv::IMREAD_GRAYSCALE);
   if (image1.empty())
   {
       std::cerr << "Error: could not open image file '" << img1_name
                 << "'." << std::endl;
       return -1;
   }
   ///Load the image2 (if it is set).
   cv::Mat image2;
   if (img2_name != "")
   {
       image2 = cv::imread(img2_name, cv::IMREAD_GRAYSCALE);
       if (image2.empty())
       {
           std::cerr << "Error: could not open image file '" << img2_name
                     << "'." << std::endl;
           return -1;
       }
   }
   else
   {
      //TODO: Flip the image over X axe.
      
      cv::flip(image1, image2, 0);
       //
   }
   
   assert(!image2.empty());

   //TODO: convert images to [0,1] float.


   image1 = fsiv_normalize_minmax(image1);
   image2 = fsiv_normalize_minmax(image2);
   //

   assert(image1.type()==CV_32FC1);
   assert(image2.type()==CV_32FC1);

   /// Compute LBP matrixs   
   cv::Mat lbpmat1, lbpmat2;

   fsiv_lbp(image1, lbpmat1, false);
   fsiv_lbp(image2, lbpmat2, false);

   /// Compute the LBP histograms
   cv::Mat lbp_desc1, lbp_desc2;

   fsiv_lbp_desc(lbpmat1, lbp_desc1, ncells, true, false);
   fsiv_lbp_desc(lbpmat2, lbp_desc2, ncells, true, false);

   /// Compute the Chi^2 distance between the input image1 and image2, if it is set,
   /// or else between image1 and its mirror.
   float dist = fsiv_chisquared_dist(lbp_desc1, lbp_desc2);
   // Show distance
   std::cout << "Distance between image1, image2 = " << dist << std::endl;

   /// Display images
   cv::Mat lbpdesc1_img;
   draw_histogram_descriptor(lbp_desc1, lbpdesc1_img, ncells);
   cv::Mat lbpdesc2_img;
   draw_histogram_descriptor(lbp_desc2, lbpdesc2_img, ncells);

   cv::imshow("Image1", image1);
   cv::imshow("Image2", image2);
   cv::imshow("LBP image1", lbpmat1);
   cv::imshow("LBP image2", lbpmat2);
   cv::imshow("LBPDESC1", lbpdesc1_img);
   cv::imshow("LBPDESC2", lbpdesc2_img);
   cv::waitKey(0);
        
   
   return 0;
}
