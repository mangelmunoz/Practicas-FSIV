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
    "{@image2        |<none>|image to show.}"
  ;

cv::Mat img_2;
int flag = 0;

cv::String img_name, img_name2;

void on_mouse(int event, int x, int y, int flags, void *userdata)
{

  //Si clicamos, guardamos las coordenadas
	if (event == cv::EVENT_LBUTTONDOWN)
	{
        //Si flag está a 0, es el primer click
        if(flag == 0){
	    	
        	flag = 1;

	    	static_cast<int*>(userdata)[0] = x;
	      	static_cast<int*>(userdata)[1] = y;
        
        }

        //Si flag está a 1, es el segundo click
        else if(flag == 1)
        {
          //Guardamos el tamaño en pixeles de la imagen
          cv::Size patata = img_2.size();

          static_cast<int*>(userdata)[2] = x;
          static_cast<int*>(userdata)[3] = y;
        
          int x_mayor, x_menor;
          int y_mayor, y_menor;
          
          //Guardamos en variables las 4 coordenadas obtenidas, de mayor a menor
          if(static_cast<int*>(userdata)[0] < x)
          {
            x_mayor = x;
            x_menor = static_cast<int*>(userdata)[0];
          }

          else
          {
            x_menor = x;
            x_mayor = static_cast<int*>(userdata)[0];
          }

          if(static_cast<int*>(userdata)[1] < y)
          {
            y_mayor = y;
            y_menor = static_cast<int*>(userdata)[1];
          }

          else
          {
            y_menor = y;
            y_mayor = static_cast<int*>(userdata)[1];
          }

          cv::Vec3b* ptr;
          int avg;

          //Recorremos la imagen
          for(int i = 0; i < img_2.cols; i++)
          {
          	ptr = img_2.ptr<cv::Vec3b>(i);

            for(int j = 0; j < img_2.rows; j++)
            {
              //Si el punto en que nos encontramos se encuentra fuera del rectángulo formado
              //por los 2 puntos, ponemos el pixel en blanco y negro
              if(((i < x_menor)||(i > x_mayor)) || ((j < y_menor)||(j > y_mayor)))
              {
                //Obtenemos la informacion de los 3 canales de ese pixel
     
                //Realizamos la conversion a monocroma
                avg = 0.1*ptr[j][0]+0.6*ptr[j][1]+0.3*ptr[j][2];

                //Introducimos en cada canal del pixel el nuevo valor
                ptr[j][0] = avg;
                ptr[j][1] = avg;
                ptr[j][2] = avg;
            
              } 
            }
          }

          flag = 2;
        }
	}

  //Si flag está a 2, se habrán realizado los 2 clicks 
	else if(flag == 2){

    //Cargamos la ventana
    cv::namedWindow(img_name2);

    //Visualizo la imagen cargada en la ventana.
    cv::imshow(img_name2, img_2);

    flag = 0;

    std::cout << "Pulsa ESC para salir sin guardar, o cualquier letra para guardar." << std::endl;
          
    int key = (cv::waitKey(0) & 0xff);
      
      // Si key != 27, se habrá pulsado cualquier tecla, y crearemos un fichero con la nueva imagen
      if (key != 27)
      {
        std::cout << "Guardando imagen..." << std::endl;
          
        cv::imwrite(img_name2, img_2);
      }

      // Si key == 27, destruimos todo sin guardar
      else
      {
        std::cout << "Saliendo sin guardar..." << std::endl;
        cv::destroyAllWindows();
      } 

	}

}

void on_mouse2(int event, int x, int y, int flags, void *userdata)
{
  //Si clicamos, guardamos las coordenadas
  if (event == cv::EVENT_LBUTTONDOWN)
        {
          flag = 1;

          static_cast<int*>(userdata)[0] = x;
          static_cast<int*>(userdata)[1] = y;
        }

  // Si movemos el raton, se mantiene a la espera de otro evento
  else if(event == cv::EVENT_MOUSEMOVE)
  {}
  //Si levantamos el raton, cogemos la segunda coordenada y empezamos
  else if(event = cv::EVENT_LBUTTONUP)
  {

    //Guardamos el tamaño en pixeles de la imagen
          cv::Size patata = img_2.size();

          static_cast<int*>(userdata)[2] = x;
          static_cast<int*>(userdata)[3] = y;
        
          int x_mayor, x_menor;
          int y_mayor, y_menor;
          
          //Guardamos en variables las 4 coordenadas obtenidas, de mayor a menor
          if(static_cast<int*>(userdata)[0] < x)
          {
            x_mayor = x;
            x_menor = static_cast<int*>(userdata)[0];
          }

          else
          {
            x_menor = x;
            x_mayor = static_cast<int*>(userdata)[0];
          }

          if(static_cast<int*>(userdata)[1] < y)
          {
            y_mayor = y;
            y_menor = static_cast<int*>(userdata)[1];
          }

          else
          {
            y_menor = y;
            y_mayor = static_cast<int*>(userdata)[1];
          }

          uchar *ptr;
          int avg;

          //Recorremos la imagen
          for(int i = 0; i < patata.width; i++)
          {
            for(int j = 0; j < patata.height; j++)
            {
              //Si el punto en que nos encontramos se encuentra fuera del rectángulo formado
              //por los 2 puntos, ponemos el pixel en blanco y negro
              if(((i < x_menor)||(i > x_mayor)) || ((j < y_menor)||(j > y_mayor)))
              {
                //Obtenemos la informacion de los 3 canales de ese pixel
                ptr=img_2.ptr<uchar>(j)+3*i; 

                //Realizamos la conversion a monocroma
                avg = 0.1*ptr[0]+0.6*ptr[1]+0.3*ptr[2];

                //Introducimos en cada canal del pixel el nuevo valor
                ptr[0] = avg;
                ptr[1] = avg;
                ptr[2] = avg;
            
              } 
            }
          }
          
    flag = 2;
        
  }

  if(flag == 2){

    //Cargamos la ventana
    cv::namedWindow(img_name2);

    //Visualizo la imagen cargada en la ventana.
    cv::imshow(img_name2, img_2);

    flag = 0;

    std::cout << "Pulsa ESC para salir sin guardar, o cualquier letra para guardar." << std::endl;
          
    int key = (cv::waitKey(0) & 0xff);

      // Si key != 27, se habrá pulsado cualquier tecla, y crearemos un fichero con la nueva imagen
      if (key != 27)
      {
        std::cout << "Guardando imagen..." << std::endl;
          
        cv::imwrite(img_name2, img_2);
      }

      // Si key == 27, destruimos todo sin guardar
      else
      {
        std::cout << "Saliendo sin guardar..." << std::endl;
        cv::destroyAllWindows();
      } 
  }

}


void menu()
{ 
  
  std::cout<<std::endl<<"INTRODUZCA UNA OPCION"<<std::endl;

  std::cout<<"Opcion 1: Selección con dos clicks"<<std::endl;
  std::cout<<"Opcion 2: Seleccion interactiva arrastrando el raton"<<std::endl;
  std::cout<<"Opcion 0: Salir"<<std::endl;


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

      img_name = parser.get<cv::String>("@image"); //Obtenermos el argumento @image.
      img_name2 = parser.get<cv::String>("@image2"); //Obtenermos el argumento @image2.

      if (!parser.check())
      {
          parser.printErrors();
          return 0;
      }

    
      int opcion = 1;

      int coords[4];
     
      //Mientras no escribamos 0 en la opcion, el programa se seguirá ejecutando
      while(opcion != 0){

      menu();
      std::cin>>opcion;

      switch(opcion)
      {
        case 1: 

          //Cargamos la imagen 
          img_2 = cv::imread(img_name, cv::IMREAD_UNCHANGED);

          if (img_2.empty())
          {
             std::cerr << "Error: no he podido abrir el fichero '" << img_name << "'." << std::endl;
             return EXIT_FAILURE;
          }
      
          //Cargamos la ventana
          cv::namedWindow(img_name);


          //Visualizo la imagen cargada en la ventana.
          cv::imshow(img_name, img_2);

          //Realizamos callback en la imagen cargada, con la funcion on_mouse
          cv::setMouseCallback (img_name, on_mouse, coords);

          //Hasta que no se pulse una tecla no saldra del programa
          cv::waitKey(0) & 0xff; 

          cv::destroyAllWindows();

        break;

        case 2:

          //Cargamos la imagen 
          img_2 = cv::imread(img_name, cv::IMREAD_UNCHANGED);

          if (img_2.empty())
          {
             std::cerr << "Error: no he podido abrir el fichero '" << img_name << "'." << std::endl;
             return EXIT_FAILURE;
          }

          //Cargamos la ventana
          cv::namedWindow(img_name);

          //Visualizo la imagen cargada en la ventana.
          cv::imshow(img_name, img_2);

          //Realizamos callback en la imagen cargada, con la funcion on_mouse2
          cv::setMouseCallback (img_name, on_mouse2, coords);
      
          //Hasta que no se pulse una tecla no saldra del programa
          cv::waitKey(0) & 0xff; 

          cv::destroyAllWindows();

        break;

        case 0: exit(-1);

        break;
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



