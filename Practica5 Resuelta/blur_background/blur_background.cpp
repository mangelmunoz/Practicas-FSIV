/*!
  Esto es un esqueleto de programa para usar en las prácticas
  de Visión Artificial.

  Se supone que se utilizará OpenCV.

  Para compilar, puedes ejecutar:
    g++ -Wall -o esqueleto esqueleto.cc `pkg-config opencv --cflags --libs`

*/

#include <iostream>
#include <exception>

//Includes para OpenCV, Descomentar según los módulo utilizados.
#include <opencv2/core/core.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/video.hpp>
#include <opencv2/calib3d/calib3d.hpp>

/**
 * @brief Compute the dense optical flow using the Farneback method.
 * @param[in] prev is the previus frame.
 * @param[in] next is the next frame.
 * @param[in,out] flow is the dense optical flow. If not is empty, it will be used as intial guess.
 * @param[in] params optional optical flow method parameters.
 * @pre prev.type()==CV_8UC1
 * @pre next.type()==CV_8UC1
 * @pre flow.empty() || flow.type()==CV_32FC2
 * @post flow.type()==CV_32FC2
 */

void fsiv_compute_dense_optical_flow(cv::Mat const& prev, cv::Mat const& next, cv::Mat & flow, void * params=0)
{
    CV_Assert(next.type()==CV_8UC1);
    CV_Assert(prev.type()==CV_8UC1);
    CV_Assert(flow.empty() || flow.type()==CV_32FC2);

    cv::calcOpticalFlowFarneback(prev, next, flow, 0.4, 1, 12, 2, 8, 1.2, 0);

    CV_Assert(flow.type()==CV_32FC2);
}

/**
 * @brief Compute the magnitude of the optical flow.
 * @param[in] flow is the dense opctical flow.
 * @param mag if the flow magnitude.
 * @pre flow.type()==CV_32FC2
 * @post mag.type()==CV_32FC1
 */
void fsiv_compute_optical_flow_magnitude(cv::Mat &flow, cv::Mat& mag)
{
    CV_Assert(flow.type()==CV_32FC2);

    std::vector<cv::Mat> channels;
    cv::Mat angle;
    
    split(flow, channels);

    cv::magnitude(channels[0], channels[1], mag);

   CV_Assert(mag.type() == CV_32FC1);
}

/**
 * @brief The OF_BBLUR_STEPS to do.
 * STEP1, Normalize OF mag to interval [0,1].
 * STEP2, Theshold to {0/1} the OF mag >= t.
 * STEP3, Dilate the mask.
 * STEP4, Adds a running average with alpha memory factor.
 *
 * You can combine steps using bitwise or. For example all the steps
 * STEP1 | STEP2 | STEP3 | STEP4
 */

enum OF_BBLUR_STEPS
{
    STEP1=1, /**< Normalize OF mag to interval [0,1].*/
    STEP2=2, /**< Theshold to {0/1} the OF mag >= t .*/
    STEP3=4, /**< Dilate the mask.*/
    STEP4=8, /**< Adds a running average with alpha memory factor.*/
};

/*
 * @brief Compute a foreground mask using the magnitude of dense optical flow as feature.
 * @see OF_BBLUR_STEPS
 * @param[in] mag is the OF magnitude.
 * @param[in,out] mask is the output mask.
 * @param[in] method specifies how to compute the mask.
 * @param[in] alpha is the memory factor.
 * @param[in] ste_r is the morphological structure element radius.
 * @param[in] t is the magnitude threshold.
 * @pre !(steps&STEP2) || t>=0.0
 * @pre !(steps&STEP3) || !ste.empty()
 * @pre !(steps&STEP4) || (!mask.empty() && mask.type()==CV_32FC1 && (0<=alpha && alpha<1))
 */
void fsiv_compute_of_foreground_mask(cv::Mat const& mag, cv::Mat & mask, const int steps, const double alpha, cv::Mat const& ste, const double t)
{
    CV_Assert(mag.type()==CV_32FC1);
    CV_Assert(!(steps&STEP2) || t>=0.0);
    CV_Assert(!(steps&STEP3) || !ste.empty());
    CV_Assert(!(steps&STEP4) || (!mask.empty() && mask.type()==CV_32FC1 && (0<=alpha && alpha<1)));

    //TODO
    //HINT: test steps to do with steps & STEPX
    //see doc of enum OF_BBLUR_STEPS to know what to do in each step.
    cv::Mat old_mask;

    mask.copyTo(old_mask);

    if(steps >= 1)
    {
    	cv::normalize(mag, mask, 0, 1, cv::NORM_MINMAX);
    }
    
    if(steps >= 2)
    {
	    cv::threshold(mag,mask,t,1, cv::THRESH_BINARY);
    }
    

    if(steps >= 4)
    {
    	cv::dilate(mask, mask, ste, cv::Point(-1,-1), 1);
    }
   
   	if(steps == 8)
   	{	
   		mask = alpha * old_mask + (1.0 - alpha) * mask;
   	}


    CV_Assert(mask.type()==CV_32FC1);
}

/**
 * @brief Blur the background of a frame given the foreground mask.
 * @param[in] frame is the input frame.
 * @param[in] foreground_mask is the foreground mask. It's assumed in range [0.0, 1.0].
 * @param[out] bblur_frame is the output frame with the background blurred.
 * @param[in] r is the blur radius.
 * @param[in] gaussian if its true used gaussian blur, else uniform blur.
 * @pre frame.type()==CV_32FC3
 * @pre foreground_mask.type()==CV_32FC3
 * @pre r>0
 * @post bblur_frame.type()==CV_32FC3
 */
void fsiv_blur_background(cv::Mat const& frame, cv::Mat const& foreground_mask, cv::Mat & bblur_frame, const int r, const bool gaussian)
{
    CV_Assert(frame.type()==CV_32FC3);
    CV_Assert(foreground_mask.type()==CV_32FC3);
    CV_Assert(r>0);


    //TODO.
    //HINT: see cv::blur() and cv::GaussianBlur() docs.
    //Use opencv aritmetic to combine float images using a mask.

    cv::Mat aux;
    cv::Mat aux2;
    cv::Mat aux3;

    std::vector<cv::Mat> channels;
     cv::Size t(r, r);

    if(gaussian == false)
    {	
    	cv::blur(frame, bblur_frame, t);
    }
    
    else
    {
    	cv::GaussianBlur(frame, bblur_frame, t, 5, 5);
    }

    aux = frame.mul(foreground_mask);

    cv::split(foreground_mask, channels);

    channels[0] = 1 - channels[0];
    channels[1] = 1 - channels[1];
    channels[2] = 1 - channels[2];

    cv::merge(channels, aux3);
    
    aux2 = bblur_frame.mul(aux3);

    cv::add(aux, aux2, bblur_frame);
	    
    CV_Assert(bblur_frame.type()==CV_32FC3);
}

const cv::String keys =
    "{help h usage ? |      | print this message   }"
    "{c              |   	| The input string is a camera idx.}"
    "{s ste          |0     | Structure element radius for morph. Default 0 means not use.}"
    "{g gaussian     |      | Use gaussian blur, else use uniform kernel.}"
    "{n normalice    |      | normalice the optical flow magnitude.}"
    "{t threshold    |-1    | Threshold to 0/1 the optical flow magnitude to get the mask. Default -1 means don't threshold.}"
    "{a alpha        |-1     | The mask has an alpha memory factor. Default -1 means don't have memory.}"
    "{r              |11    | Blur radius. Default 11.}"
    "{@input         |      | input stream (filename or camera idx)}"
    ;

int
main (int argc, char* const* argv)
{
    int retCode=EXIT_SUCCESS;

    try
    {

        cv::CommandLineParser parser(argc, argv, keys);
        parser.about("Blur the background of the input stream using optical flow magnitued as clue.");
        if (parser.has("help"))
        {
            parser.printMessage();
            return EXIT_SUCCESS;
        }
        bool is_camera =false;
        //is_camera = parser.has("c");
        int blur_r = parser.get<int>("r");

        cv::Mat ste = cv::Mat(1,1, CV_32FC1);
        int ste_r = parser.get<int>("s");

        //TODO: if ste_r>0 assing to ste a structuring element with size 2*ste_r+1.
        //Choose the shape of the element taking into account the problem.
        //HINT: see cv::getStructuringElement() docs.
        if (ste_r>0)
        	
        	ste = cv::getStructuringElement(cv::MORPH_RECT,cv::Size(2 * ste_r + 1,2 * ste_r + 1));

            //cv::resize(ste, ste,cv::Size(2 * ste_r + 1, 2 * ste_r + 1) );
        

        CV_Assert(ste_r==0 || !ste.empty());

        bool use_gaussian_blur = parser.has("g");
        bool normalice_of_mag = parser.has("n");
        double th = parser.get<double>("t");
        double alpha = parser.get<double>("a");

        int camera_idx = 0;
        std::string input_file;
        
        if (is_camera)
            camera_idx = parser.get<int>("c");
        else
            input_file = parser.get<std::string>("@input");
        if (!parser.check())
        {
            parser.printErrors();
            return EXIT_FAILURE;
        }
        
        cv::VideoCapture cap;
        
        //TODO: regarding is_camare, open the input video stream according.
        //See cv::VideoCapture docs.
        if (is_camera)
        {
            cap.open(camera_idx);
        }
        else
        {
            cap.open(input_file);
        }


        if (!cap.isOpened())
        {
            std::cerr << "Error: could not open the input stream!" << std::endl;
            return EXIT_FAILURE;
        }

         int STEPS=STEP1;
        //TODO: Set the steps for the algorithm to compute the foreground mask in
        // the STEPS variable.
        // HINT: the default values for the CLI params: normalice_of_mag,
        // th, ste and alpha say if a step is used or not.
         if(normalice_of_mag == true)
         {
         	STEPS = STEP1;

         	if(th != -1)
         	{
	         	STEPS=STEP2;

	         	if(ste_r != 0)
	         	{
	         		STEPS=STEP3;

	         		if(alpha != -1)
	         		{

	         			STEPS=STEP4;
	         		}
	         	}
	         }
         }

       CV_Assert(STEPS>0);

        cv::Mat frame1;
        cap.read(frame1);
        cv::Mat prev;
        cv::cvtColor(frame1, prev, cv::COLOR_BGR2GRAY);

        cv::Mat frame2;

        int key=0;
        cv::Mat foreg_mask = cv::Mat::ones(frame1.rows, frame1.cols, CV_32FC1);

        cv::Mat next;
        cv::Mat frame2_f;
        cv::Mat flow;
        cv::Mat mag;
        std::vector<cv::Mat> mask_channels(3);
        cv::Mat foreg_mask_rgb;
        cv::Mat blur_img;
        cv::Mat rgb;

        while ((key!=27) && cap.read(frame2))
        {            
            cv::cvtColor(frame2, next, cv::COLOR_BGR2GRAY);            
            frame2.convertTo(frame2_f, CV_32F, 1.0/255.0);

            fsiv_compute_dense_optical_flow(prev, next, flow);

            fsiv_compute_optical_flow_magnitude(flow, mag);

            cv::imshow("FLOW MAGNITUDE", mag);

            fsiv_compute_of_foreground_mask(mag, foreg_mask, STEPS, alpha, ste, th);

            split(foreg_mask, mask_channels);

            // //TODO: Interlace the mask to get a 3 channels mask.
            // //Hint: use the mask_channels vector.

            cv::cvtColor(foreg_mask, foreg_mask_rgb, cv::COLOR_GRAY2BGR);   
            
            // //
            CV_Assert(foreg_mask_rgb.type()==CV_32FC3);

             cv::imshow("MASK", foreg_mask_rgb);

            fsiv_blur_background(frame2_f, foreg_mask_rgb, rgb, blur_r, use_gaussian_blur);

            cv::imshow("Original", frame2);
            cv::imshow("Blurred background", rgb);

            next.copyTo(prev);
            key = cv::waitKey(20) & 0xff;
        }
        cv::destroyAllWindows();
    }
    catch (std::exception& e)
    {
        std::cerr << "Capturada excepcion: " << e.what() << std::endl;
        retCode = EXIT_FAILURE;
    }
    return retCode;
}
