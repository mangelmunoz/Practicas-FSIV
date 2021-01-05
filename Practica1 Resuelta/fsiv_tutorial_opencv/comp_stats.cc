/*!
  Ejemplo de carga de imagen y su recorrido para calcular estadisticos
  simples.

  Se supone que se utilizará OpenCV.

  Para compilar, puedes ejecutar:
    g++ -Wall -o esqueleto esqueleto.cc `pkg-config opencv --cflags --libs`

*/

#include <iostream>
#include <exception>
#include <valarray>

//Includes para OpenCV, Descomentar según los módulo utilizados.
#include <opencv2/core/core.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
//#include <opencv2/calib3d/calib3d.hpp>

const cv::String keys =
    "{help h usage ? |      | print this message.   }"
    "{@image         |<none>| input image.          }"            
    ;

/*!
    @brief Calcular el valor medio de una imagen y su varianza.
    
    Ejemplo simple de recorrido de una imagen. OpenCv ya tiene funciones
    para calcular los valores medios y las desviaciones de forma más eficiente.
    
    Esta es la forma menos eficiente de recorrer una imagen. Mira la
    documentacion de opencv para ver otras.
    
    @param[in] img es la imagen de entrada.
    @param[out] media la media de los valores.
    @param[out] var la varianza de los valores.
    
    @pre img no está vacia.
    @pre img es de tipo CV_8UC1 (Un sólo canal en formato byte).
*/    
void 
compute_stats(const cv::Mat& img, float& media, float& var)
{
    //Comprobacion de precondiciones.
    CV_Assert( !img.empty() );
    CV_Assert( img.type() == CV_8UC1 );
   
    media = 0.0;
    var = 0.0;
    
    //Para cada fila 0 ... img.rows-1
    for (int row=0; row<img.rows; ++row)
        //Para cada columna 0 ... img.cols-1
        for (int col=0; col<img.cols; ++col)
        {
            //Acceder a un pixel con el metodo at<Tipo de pixel>(fila, columna).
            //Aqui cada pixel es un byte (uchar).

            const uchar v = img.at<uchar>(row, col);

            //También podríamos tener otros tipos:
            //   Tres bytes por pixel (CV_8UC3) ->  at<cv::Vec3b>
            //   Un solo float por pixel (CV_32FC1)  ->  at<float>
            //   tres floats por pixel (CV_32FC3)   -> at<cv::Vec3f>
            //   ... mas combinaciones.


            media += v;
            var += v*v;
        }
    const float count = img.rows*img.cols;
    
    media /= count;
    var /= count;
    var = var - media*media;
       
}

int
main (int argc, char* const* argv)
{
  int retCode=EXIT_SUCCESS;
  
  try {    

      cv::CommandLineParser parser(argc, argv, keys);
      parser.about("Calcula los estadísticos de una imagen v1.0.0");

      if (parser.has("help"))
      {
          parser.printMessage();
          return 0;
      }
      cv::String img_name = parser.get<cv::String>("@image");

      if (!parser.check())
      {
          parser.printErrors();
          return 0;
      }


      //Carga la imagen desde archivo.
      //En funcion de como se compilo opencv podra
      //cargar mas o menos formatos graficos.
      //Lee la documentacion de imread para ver mas detalles.
      cv::Mat img = cv::imread(img_name, cv::IMREAD_UNCHANGED);
      
      if (img.empty())
      {
         std::cerr << "Error: no he podido abrir el fichero '" << img_name << "'." << std::endl;
         return EXIT_FAILURE;
      }
        
        std::vector<cv::Mat> canales;
        cv::split(img, canales); //De-entrelaza la imagen (si lo está), guardando cada canal como una matriz diferente).
        for(size_t c = 0; c<canales.size(); ++c)
        {
            float media = 0.0f;
            float var = 0.0f;
            compute_stats(canales[c], media, var);
            std::cout << "Canal " << c << " media: " << media << " desviación: " << sqrt(var) << std::endl;
        }


      //Creo la ventana grafica para visualizar la imagen.
      //El nombre de la ventana sirve como 'handle' para gestionarla despues.
      //Lee la documentacon de namedWindow para mas detalles.
      cv::namedWindow("IMG");

      //Visualizo la imagen cargada en la ventana.
      cv::imshow("IMG", img);

      //Para que se actualice el la interfaz grafica es necesario llamar a
      //waitKey. Además esta función nos permite interactuar con el teclado.
      //Lee la documentacion de waitKey para mas detalles.
      cv::waitKey(0);
   
  }
  catch (std::exception& e)
  {
    std::cerr << "Capturada excepcion: " << e.what() << std::endl;
    retCode = EXIT_FAILURE;
  }
  return retCode;
}




