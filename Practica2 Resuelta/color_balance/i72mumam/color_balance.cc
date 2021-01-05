/*!
  Esto es un esqueleto de programa para usar en las prácticas
  de Visión Artificial.

  Se supone que se utilizará OpenCV.

  Para compilar, puedes ejecutar:
    g++ -Wall -o esqueleto esqueleto.cc `pkg-config opencv --cflags --libs`

*/

#include <iostream>
#include <exception>
#include <vector>

//Includes para OpenCV, Descomentar según los módulo utilizados.
#include <opencv2/core/core.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
//#include <opencv2/calib3d/calib3d.hpp>

const cv::String keys =
    "{help h usage ? |      | print this message   }"
    "{p              |<none>     | Percentage of brightest points used. Default 0 means use the most.}"
    "{@input         |<none>| input image.}"
    "{@output        |<none>| output image.}"
    ;


/**
 * @brief Apply a "white patch" color balance operation to the image.
 * @arg[in] in is the imput image.
 * @return the color balanced image.
 * @pre in.type()==CV_8UC3
 * @warning A BGR color space is assumed for the input image.
 */
cv::Mat fsiv_wp_color_balance(cv::Mat in)
{
    CV_Assert(in.type()==CV_8UC3);


    
    //1º Calculamos cual es el pixel con mayor intensidad, y guardamos sus valores RGB
    // Intensidad = (0.6R* 0.3G * 0.1B)/3

    // 2º Transformamos la imagen
    // Transformacion = (R * 255)/Rmasluminoso; (G * 255)/Gmasluminoso; (B * 255)/Gmasluminoso
    
    
    
    float intensidad = 0, intensidadpixel = 0, rojo, azul, verde;
    cv::Vec3b ptr;

    for(int i = 0; i < in.rows; i++)
    {
    	for(int j = 0; j < in.cols; j++)
    	{
    		ptr=in.at<cv::Vec3b>(i,j);

    		intensidadpixel = (0.1*ptr[0]+0.6*ptr[1]+0.3*ptr[2]) / 3;
	
    		if(intensidadpixel >  intensidad)
    		{

    			intensidad = intensidadpixel;
    			
    			rojo = in.at<cv::Vec3b>(i,j)[2];
    			verde = in.at<cv::Vec3b>(i,j)[1];
    			azul = in.at<cv::Vec3b>(i,j)[0];
    		}
    	}
    }

    cv::Mat in2 = in;

    for(int i = 0; i < in2.rows; i++)
    {
    	for(int j = 0; j < in2.cols; j++)
    	{
    		ptr=in.at<cv::Vec3b>(i,j);

    		ptr[0] = (ptr[0] * 255) / azul;
    		ptr[1] = (ptr[1] * 255) / verde;
    		ptr[2] = (ptr[2] * 255) / rojo;

    		in2.at<cv::Vec3b>(i,j) = ptr;
    	}
    }

    return in2;
}

/**
 * @brief Apply a "gray world" color balance operation to the image.
 * @arg[in] in is the imput image.
 * @arg[in] p is the percentage of brightest points used to calculate the color correction factor.
 * @return the color balanced image.
 * @pre in.type()==CV_8UC3
 * @pre 0.0 < p <= 100.0
 * @warning A BGR color space is assumed for the input image.
 */
cv::Mat fsiv_gw_color_balance(cv::Mat const& in, float p)
{
    CV_Assert(in.type()==CV_8UC3);
    CV_Assert(0.0f<p && p<=100.0f);

    float rojo = 0, azul = 0, verde = 0, mediarojo = 0, mediaverde = 0, mediaazul = 0;
    cv::Vec3b ptr;
    int contador = 0;

    cv::Mat in2;
	    
	    if(p == 100)
	    {
		    for(int i = 0; i < in.rows; i++)
		    {
		    	for(int j = 0; j < in.cols; j++)
		    	{
		    		ptr=in.at<cv::Vec3b>(i,j);

		    		rojo += in.at<cv::Vec3b>(i,j)[2];
		    		verde += in.at<cv::Vec3b>(i,j)[1];
		    		azul += in.at<cv::Vec3b>(i,j)[0];

		    		contador++;
		    	}
		    }

		    mediarojo = rojo / contador;
		    mediaverde = verde / contador;
		    mediaazul = azul / contador;
		    in2 = in;

		    for(int i = 0; i < in2.rows; i++)
		    {
		    	for(int j = 0; j < in2.cols; j++)
		    	{
		    		ptr=in.at<cv::Vec3b>(i,j);

		    		ptr[0] = (ptr[0] * 128) / mediaazul;
		    		ptr[1] = (ptr[1] * 128) / mediaverde;
		    		ptr[2] = (ptr[2] * 128) / mediarojo;

		    		

		    		in2.at<cv::Vec3b>(i,j) = ptr;
		    	}
		    }
	    }


    return in2;
}

/**
 * @brief Application State.
 * Use this structure to maintain the state of the application
 * that will be passed to the callbacks.
 */
struct UserData
{
    //TODO

};

/** @brief Standard mouse callback
 * Use this function an argument for cv::setMouseCallback to control the
 * mouse interaction with a window.
 *
 * @arg event says which mouse event (move, push/release a mouse button ...)
 * @arg x and
 * @arg y says where the mouse is.
 * @arg flags give some keyboard state.
 * @arg user_data allow to pass user data to the callback.
 */
void on_mouse (int event, int x, int y, int flags, void * user_data_)
{
    UserData *user_data = static_cast<UserData*>(user_data_);
    if (event == cv::EVENT_LBUTTONDOWN)
    {
        //TODO

    }
}

/** @brief Standard trackbar callback
 * Use this function an argument for cv::createTrackbar to control
 * the trackbar changes.
 *
 * @arg v give the trackbar position.
 * @arg user_data allow to pass user data to the callback.
 */
void on_change(int v, void * user_data_)
{
    UserData * user_data = static_cast<UserData*>(user_data_);

    //TODO
}

int
main (int argc, char* const* argv)
{
    int retCode=EXIT_SUCCESS;

    try {

        cv::CommandLineParser parser(argc, argv, keys);
        parser.about("Apply a color balance to an image.");
        if (parser.has("help"))
        {
            parser.printMessage();
            return EXIT_SUCCESS;
        }
        int p = parser.get<int>("p");
        if (p<0 || p>100)
        {
            std::cerr << "Error: p is out of range [0, 100]." << std::endl;
            return EXIT_FAILURE;
        }
        cv::String input_n = parser.get<cv::String>("@input");
        cv::String output_n = parser.get<cv::String>("@output");

        if (!parser.check())
        {
            parser.printErrors();
            return EXIT_FAILURE;
        }


        cv::Mat img = cv::imread(input_n, cv::IMREAD_UNCHANGED);

          if (img.empty())
          {
             std::cerr << "Error: no he podido abrir el fichero '" << input_n << "'." << std::endl;
             return EXIT_FAILURE;
          }


         cv::namedWindow(input_n);
	        cv::imshow(input_n, img);

        if(p == 0)
        {

	        cv::Mat img_2 = fsiv_wp_color_balance(img);

	        cv::namedWindow(output_n);
        
	       	cv::imshow(output_n, img_2);

	       	std::cout << "Pulsa ESC para salir sin guardar, o cualquier letra para guardar." << std::endl;
          
    		int key = (cv::waitKey(0) & 0xff);
      
		      // Si key != 27, se habrá pulsado cualquier tecla, y crearemos un fichero con la nueva imagen
		      if (key == 13)
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
        else
        {
        	cv::Mat img_2 = fsiv_gw_color_balance(img, p);

	        cv::namedWindow(output_n);
       		
       		cv::imshow(output_n, img_2);

       		std::cout << "Pulsa ESC para salir sin guardar, o cualquier letra para guardar." << std::endl;

       		int key = (cv::waitKey(0) & 0xff);
      
		      // Si key != 27, se habrá pulsado cualquier tecla, y crearemos un fichero con la nueva imagen
		      if (key == 13)
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
    }
    catch (std::exception& e)
    {
        std::cerr << "Capturada excepcion: " << e.what() << std::endl;
        retCode = EXIT_FAILURE;
    }
    return retCode;
}
