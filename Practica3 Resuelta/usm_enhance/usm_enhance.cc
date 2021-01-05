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
//#include <opencv2/calib3d/calib3d.hpp>

const cv::String keys =
    "{help h usage ? |      | print this message   }"
    "{r              |1     | Window's radius. Default 1.}"
    "{g              |1.0   | Enhance's gain. Default 1.0}"
    "{c              |      | Use circular convolution.}"
    "{f              |0     | Filter type: 0->Box, 1->Gaussian. Default 0.}"
    "{@input         |<none>| input image.}"
    "{@output        |<none>| output image.}"
    ;


/**
 * @brief Return a box filter.
 * @arg[in] r is the filter's radius.
 * @pre r>0;
 * @post retV.rows==retV.cols==2*r+1
 * @post cv::sum(retV)==1.0
 */

cv::Mat fsiv_create_box_filter(const float r)
{
    CV_Assert(r>0);

 
    cv::Mat M(2*r+1,2*r+1, CV_32FC1);

    M = 1/((2*r+1) * (2*r + 1));
    

    // for(int i = 0; i < M.rows; i++)
    // {
    // 	for(int j = 0; j < M.cols; j++)
    // 	{
    	
    // 		M.at<float>(i,j) = 1/((2*r+1) * (2*r + 1));
    // 	}
    // }



    CV_Assert(M.rows==2*r+1);
    CV_Assert(M.cols==2*r+1);
    //CV_Assert(cv::sum(M)[0]== 1);

    return M;
}

/**
 * @brief Return a Gaussian filter.
 * @arg[in] r is the filter's radius.
 * @pre r>0;
 * @post retV.rows==retV.cols==2*r+1
 * @post cv::sum(retV)==1.0
 */
cv::Mat fsiv_create_gaussian_filter(const int r)
{
    CV_Assert(r>0);

    //TODO

    return cv::Mat();
}

/**
 * @brief Expand an image with zero padding.
 * @arg[in] in is the input image.
 * @arg[in] r is the window's radius to expand.
 * @return the expanded image.
 * @pre !in.empty()
 * @pre r>0
 * @post retV.type()==in.type()
 * @post retV.rows==in.rows+2*r
 * @post retV.cols==in.cols+2*r
 */
cv::Mat fsiv_fill_expansion(cv::Mat const& in, const int r)
{
    CV_Assert(!in.empty());
    CV_Assert(r>0);

    cv::Mat expansion;
    expansion = cv::Mat::zeros(in.rows+2*r,in.cols+2*r, CV_32FC1);

	    for(int i = r; i < expansion.rows - r; i++)
	    {
	    	for(int j = r; j < expansion.cols - r; j++)
	    	{
	    		expansion.at<float>(i,j) = in.at<float>(i - r,j - r);	
	    	}
	    }

    //CV_Assert(expansion.type()==in.type());
    CV_Assert(expansion.rows==in.rows+2*r);
    CV_Assert(expansion.cols==in.cols+2*r);

    return expansion;
}

/**
 * @brief Circular expansion of an image.
 * @arg[in] in is the input image.
 * @arg[in] r is the window's radius to expand.
 * @return the expanded image.
 * @pre !in.empty()
 * @pre in.type()==CV_32FC1
 * @pre r>0
 * @post retV.type()==in.type()
 * @post retV.rows==in.rows+2*r
 * @post retV.cols==in.cols+2*r
 */
cv::Mat fsiv_circular_expansion(cv::Mat const& in, const int r)
{
    CV_Assert(!in.empty());
    CV_Assert(in.type()==CV_32FC1);
    CV_Assert(r>0);


    return cv::Mat();
}
/**
 * @brief Compute the digital convolution.
 * By default the input image is expanded with zero padding.
 * Note: Code from scracth. Use cv::filter2D() is not allowed.
 * @arg[in] in is the input image.
 * @arg[in] filter is the filter to be applied.
 * @arg[in] circular if it is true, the input image is expanded following a circular fashion.
 * @pre !in.empty() && !filter.empty()
 * @pre (filter.rows % 2) >0
 * @pre (filter.cols % 2) >0
 * @pre in.type()==CV_32FC1 && filter.type()==CV_32FC1.
 * @post ret.type()==CV_32FC1 && ret.size == in.size.
 */
cv::Mat fsiv_convolve(cv::Mat const& in, cv::Mat const& filter, const bool circular=false)
{
    CV_Assert(!in.empty() && !filter.empty());
    CV_Assert((filter.rows%2 >0 ) && (filter.cols%2 >0));
    CV_Assert(in.type()==CV_32FC1);
    CV_Assert(filter.type()==CV_32FC1);

    cv::Mat salida(in.rows, in.cols, CV_32FC1);
    cv::Mat aux=in.clone();
    cv::Mat aux2=filter.clone();

    float total = 0;

    int r = (filter.rows-1)/2;
    int x = 0, y = 0;

    salida = fsiv_fill_expansion(in, r);

    	for(int i = r; i < in.rows - r; i++)
    	{
            float * salidaPtr = salida.ptr<float>(i);

    		for(int j = r; j < in.cols - r; j++)
    		{
                x = 0;
                
                for (int k = i-r; k <= i+r; k++, x++)
                {
                    
                    float * inPtr = aux.ptr<float>(k);
                    float * filterPtr = aux2.ptr<float>(x);
                    y = 0;

                        for (int l=j-r; l <= j+r; l++, y++)
                        {
                            
                            total += inPtr[l] * filterPtr[y];
                            
                        }
                }
                
                salidaPtr[j] = total;
                total = 0;

    	}   			
	
    }

    return salida;
}

/**
 * @brief Apply an unsharp mask enhance to the input image.
 * @arg[in] in is the input image.
 * @arg[in] g is the enhance's gain.
 * @arg[in] r is the window's radius.
 * @arg[in] filter_type specifies which filter to use. 0->Box, 1->Gaussian.
 * @arg[in] circular specifies if it is true, it be used circular expansion to do the convolution, else it is used zero padding.
 * @pre !in.empty()
 * @pre in.type()==CV_32FC1
 * @pre g>=0.0
 * @pre r>0
 * @pre filter_type is {0, 1}
 */
cv::Mat fsiv_usm_enhance(cv::Mat  const& in, float g=1.0, int r=1,
                         int filter_type=0, bool circular=false)
{
    CV_Assert(!in.empty());
    //CV_Assert(in.type()==CV_32FC1);
    CV_Assert(r>0);
    CV_Assert(g>=0.0);

    cv::Mat filter;

    if(filter_type == 0)
    {
    	filter = fsiv_create_box_filter(r);
    }

    cv::Mat expansion = fsiv_fill_expansion(in, r);
    cv::Mat salida(expansion.rows, expansion.cols, CV_32FC1);
    
    // std::cout<<expansion<<std::endl;

    cv::Mat convolution = fsiv_convolve(expansion, filter, false);

    cv::Mat resultado = (g + 1)*expansion - (g*convolution);

    for(int i = 0; i < resultado.rows; i++)
    {
        for(int j = 0; j < resultado.cols; j++)
        {
            if(resultado.at<float>(i,j) != 0)
            {
                salida.at<float>(i,j) = resultado.at<float>(i,j);
            }
        }
    }

    return resultado;
}

/**
 * @brief Application State.
 * Use this structure to maintain the state of the application
 * that will be passed to the callbacks.
 */
struct UserData
{
    //Use this type if you implement a GUI.
};

/** @brief Standard trackbar callback
 * Use this function an argument for cv::createTrackbar to control
 * the trackbar changes.
 *
 * @arg v give the trackbar position.
 * @arg user_data allow to pass user data to the callback.
 */
void on_change_r(int v, void * user_data_)
{
    UserData * user_data = static_cast<UserData*>(user_data_);
    //Use this callback if you implement a GUI.
}

/**
 * @brief Standard trackbar callback
 * Use this function an argument for cv::createTrackbar to control
 * the trackbar changes.
 *
 * @arg v give the trackbar position.
 * @arg user_data allow to pass user data to the callback.
 */
void on_change_g(int v, void * user_data_)
{
    UserData * user_data = static_cast<UserData*>(user_data_);
    //Use this callback if you implement a GUI.
}

/**
 * @brief Standard trackbar callback
 * Use this function an argument for cv::createTrackbar to control
 * the trackbar changes.
 *
 * @arg v give the trackbar position.
 * @arg user_data allow to pass user data to the callback.
 */
void on_change_f(int v, void * user_data_)
{
    UserData * user_data = static_cast<UserData*>(user_data_);
    //Use this callback if you implement a GUI.
}

/**
 * @brief Standard button callback
 * Use this function an argument for cv::createTrackbar to control
 * the trackbar changes.
 *
 * @arg state give the button's state.
 * @arg user_data allow to pass user data to the callback.
 */
void on_change_c(int state, void *user_data_)
{
    UserData * user_data = static_cast<UserData*>(user_data_);
    //Use this callback if you implement a GUI.
}

int
main (int argc, char* const* argv)
{
    int retCode=EXIT_SUCCESS;

    try {
        cv::CommandLineParser parser(argc, argv, keys);
        parser.about("Apply an unsharp mask enhance to an image.");
        if (parser.has("help"))
        {
            parser.printMessage();
            return EXIT_SUCCESS;
        }

        //TODO
        //manage the CLI parameters.

        cv::String input_n = parser.get<cv::String>("@input");
        cv::String output_n = parser.get<cv::String>("@output");
        if (!parser.check())
        {
            parser.printErrors();
            return EXIT_FAILURE;
        }

        float g = parser.get<float>("g");
        float r = parser.get<float>("r");

        cv::Mat in = cv::imread(input_n, cv::IMREAD_UNCHANGED);
        if (in.empty())
        {
            std::cerr << "Error: could not open input image '" << input_n
                      << "'." << std::endl;
            return EXIT_FAILURE;
        }
        cv::namedWindow(input_n);
	        cv::imshow(input_n, in);

        in.convertTo(in, CV_32FC1);
        in*= 1./255;

         cv::Mat img_2 = fsiv_usm_enhance(in, g, r, 0, false);

	        cv::namedWindow(output_n);
        
	       	cv::imshow(output_n, img_2);

	       	std::cout << "Pulsa ESC para salir sin guardar, o cualquier letra para guardar." << std::endl;
          
    		int key = (cv::waitKey(0) & 0xff);
      
		      // Si key != 27, se habrá pulsado cualquier tecla, y crearemos un fichero con la nueva imagen
		      if (key != 27)
		      {
		        std::cout << "Guardando imagen..." << std::endl;
		          
		        cv::imwrite(output_n, img_2);
		      }

		      // Si key == 27, destruimos todo sin guardar
		      else if(key == 27)
		      {
		        std::cout << "Saliendo sin guardar..." << std::endl;
		        cv::destroyAllWindows();
		      } 



    }
    catch (std::exception& e)
    {
        std::cerr << "Capturada excepcion: " << e.what() << std::endl;
        retCode = EXIT_FAILURE;
    }
    return retCode;
}
