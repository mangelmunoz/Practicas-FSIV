#include <iostream>
#include <exception>

//Includes para OpenCV, Descomentar según los módulo utilizados.
#include <opencv2/core/core.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <vector>
//#include <opencv2/calib3d/calib3d.hpp>

const cv::String keys =
    "{help h usage ? |      | print this message   }"
    "{@image         |<none>|image to show.}" //argumento sin valor por defecto.
  ;

void on_mouse(int event, int x, int y, int flags, void *userdata)
{

	// int counter = 0;
	// int pasador = 0;
	// do
	// {
	    // if (event == cv::EVENT_LBUTTONDOWN)
	    // {
	    //     static_cast<int*>(userdata)[0] = x;
	    //     static_cast<int*>(userdata)[1] = y;
	    //     // pasador+=2;
	    //     // counter++;
	    // }
	// }
	// while (counter != 2);

	cv::Mat imagen = cv::imread("ciclista_original.jpg");

	int z, t;

	// do
	// {

	if (event == cv::EVENT_LBUTTONDOWN)
	    {
	    	flags = 1;

	    	static_cast<int*>(userdata)[0] = x;
	        static_cast<int*>(userdata)[1] = y;
	    }
	else if(event == cv::EVENT_MOUSEMOVE)
	{
        // if(flags == 1)
        // {
        // 	// rectangle(imagen, cv::Point(z, t), cv::Point(x, y), cv::Scalar (0,0,0));
        // 	// 
        // }
	}
	
	else if(event = cv::EVENT_LBUTTONUP)
	{

		flags = 2;
		rectangle(imagen, cv::Point(static_cast<int*>(userdata)[0], static_cast<int*>(userdata)[1]), cv::Point(x, y), cv::Scalar (0,0,0));
	}

	
	if(flags == 2){

	cv::namedWindow("imagen_salida");

      //Visualizo la imagen cargada en la ventana.
      cv::imshow("imagen_salida", imagen);
	}
	

	

}

int main (int argc, char* const* argv)
{

  int retCode=EXIT_SUCCESS;
  
  try {    

      cv::CommandLineParser parser(argc, argv, keys);
      parser.about("realzar_primer_plano v1.0.0");
      if (parser.has("help"))
      {
          parser.printMessage();
          return 0;
      }

      cv::String img_name = parser.get<cv::String>("@image"); //Obtenermos el argumento @image1.
 
      if (!parser.check())
      {
          parser.printErrors();
          return 0;
      }

      //Cargamos la imagen
      cv::Mat img = cv::imread(img_name, cv::IMREAD_UNCHANGED);

      if (img.empty())
      {
         std::cerr << "Error: no he podido abrir el fichero '" << img_name << "'." << std::endl;
         return EXIT_FAILURE;
      }

      cv::namedWindow("imagen_entrada");

      //Visualizo la imagen cargada en la ventana.
      cv::imshow("imagen_entrada", img);

      // cv::Mat frame;

      // img >> frame;

      int coords[2], coords1[2];

      // // cv::setMouseCallback ("imagen_entrada", on_mouse, pt1);

      cv::setMouseCallback ("imagen_entrada", on_mouse, coords);
      cv::setMouseCallback ("imagen_entrada", on_mouse, coords1);



      int x = 50;
      int y = 50;
      int width = 100;
      int height = 200;

      
      cv::Mat img2;

      img2 = img;

      std::vector<cv::Mat> canales;

      cv::split(img2, canales);


      cv::Mat cambio = 0.1*canales[0] + 0.6*canales[1] + 0.3*canales[2];

      for(int i = x; i < width; i++)
      {
      	for(int j = y; j < height; j++)
      	{
      		uchar color = img.at<uchar>(cv::Point(j, i));

      		// std::cout<<color<<std::endl;

      		cambio.at<uchar>(i, j) = color;
      	}
      }
      
      // std::cout<<coords[0] coords[1], coords[2], coords[3]
      cv::Rect rect(x, height, y, width);

      // cv::cvtColor(img2, img2, cv::COLOR_RGB2GRAY);

      cv::rectangle(img2, rect, cv::COLOR_RGB2GRAY);

      // cv::imwrite("myImageWithRect.jpg",img2);

      // cv::namedWindow("imagen_salida");

      // //Visualizo la imagen cargada en la ventana.
      // cv::imshow("imagen_salida", cambio);

      // const cv::Vec3b v = img.at<cv::Vec3b>(coords[1], coords[0]);
      //    std::cout << "RGB point (" << coords[0] << ',' << coords[1] << "): "
      //              << static_cast<int>(v[0]) << ", "
      //              << static_cast<int>(v[1]) << ", "
      //              << static_cast<int>(v[2]) << std::endl;


      //CAMBIAR 101 A 27 ÑKFALÑKDÑJDAJLKAJLÑDJLADJLF
      std::cout << "Pulsa ESC para salir." << std::endl;
      while ((cv::waitKey(0) & 0xff) != 101); //Hasta que no se pulse la tecla ESC no salimos.

     }
     

	 

  catch (std::exception& e)
  {
    std::cerr << "Capturada excepcion: " << e.what() << std::endl;
    retCode = EXIT_FAILURE;
  }
  return retCode;
}
