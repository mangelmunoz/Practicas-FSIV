#include <opencv2/opencv.hpp>

#include <dlib/opencv.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
#include <dlib/gui_widgets.h>

using namespace dlib;
using namespace std;
using namespace cv;

int main(void)
{
    try
    {
        // captura imagenes de la camara o archivo de video
        cv::VideoCapture cap(0);
        if (!cap.isOpened()) {
            return 1;
        }

        // clasificador en cascada que detecta rostros frontales
        cv::CascadeClassifier face_cascade;
        face_cascade.load("files/data/haarcascade_frontalface_alt.xml");

        // crear una ventana dlib
        image_window win;
        win.set_size(420, 380);
        win.set_title("Estimar Postura 3D :: Tutor de Programacion");

        // detecta los landmarks
        shape_predictor pose_model;
        deserialize("files/data/shape_predictor_68_face_landmarks.dat") >> pose_model;

        const rgb_pixel color(255, 255, 0);
        const cv::Scalar scalar(255, 255, 0);

        std::vector<cv::Point3d> object_pts;
        object_pts.push_back(cv::Point3d(6.825897, 6.760612, 4.402142));     //#33 left brow left corner
        object_pts.push_back(cv::Point3d(1.330353, 7.122144, 6.903745));     //#29 left brow right corner
        object_pts.push_back(cv::Point3d(-1.330353, 7.122144, 6.903745));    //#34 right brow left corner
        object_pts.push_back(cv::Point3d(-6.825897, 6.760612, 4.402142));    //#38 right brow right corner
        object_pts.push_back(cv::Point3d(5.311432, 5.485328, 3.987654));     //#13 left eye left corner
        object_pts.push_back(cv::Point3d(1.789930, 5.393625, 4.413414));     //#17 left eye right corner
        object_pts.push_back(cv::Point3d(-1.789930, 5.393625, 4.413414));    //#25 right eye left corner
        object_pts.push_back(cv::Point3d(-5.311432, 5.485328, 3.987654));    //#21 right eye right corner
        object_pts.push_back(cv::Point3d(2.005628, 1.409845, 6.165652));     //#55 nose left corner
        object_pts.push_back(cv::Point3d(-2.005628, 1.409845, 6.165652));    //#49 nose right corner
        object_pts.push_back(cv::Point3d(2.774015, -2.080775, 5.048531));    //#43 mouth left corner
        object_pts.push_back(cv::Point3d(-2.774015, -2.080775, 5.048531));   //#39 mouth right corner
        object_pts.push_back(cv::Point3d(0.000000, -3.116408, 6.097667));    //#45 mouth central bottom corner
        object_pts.push_back(cv::Point3d(0.000000, -7.415691, 4.070434));    //#6 chin corner

        //2D ref points(image coordinates), referenced from detected facial feature
        std::vector<cv::Point2d> image_pts;

        //result
        cv::Mat rotation_vec;                           //3 x 1
        cv::Mat rotation_mat;                           //3 x 3 R
        cv::Mat translation_vec;                        //3 x 1 T

        

        cv::Mat pose_mat = cv::Mat(3, 4, CV_64FC1);
        cv::Mat euler_angle = cv::Mat(3, 1, CV_64FC1);

        //temp buf for decomposeProjectionMatrix()
        cv::Mat out_intrinsics = cv::Mat(3, 3, CV_64FC1);
        cv::Mat out_rotation = cv::Mat(3, 3, CV_64FC1);
        cv::Mat out_translation = cv::Mat(3, 1, CV_64FC1);

        // Grab and process frames until the main window is closed by the user.
        while (!win.is_closed())
        {
            cv::Mat temp, frame_gray;

            // omitir si no se ha podido leer la camara
            if (!cap.read(temp)) {
                continue;
            }

            // convertir a grises y equalizar histograma
            cv::cvtColor(temp, frame_gray, CV_BGR2GRAY);
            cv::equalizeHist(frame_gray, frame_gray);

            std::vector<cv::Rect> faces;
            std::vector<dlib::rectangle> facesRect;

            // detectar los rostros 
            face_cascade.detectMultiScale(frame_gray, faces, 1.1, 3, cv::CASCADE_FIND_BIGGEST_OBJECT, cv::Size(50, 50));

            // guardar la region en donde se encuentra la cara
            for (cv::Rect& rc : faces) {
                facesRect.push_back(dlib::rectangle(rc.x, rc.y, rc.x + rc.width, rc.y + rc.height));
            }

            cv_image<bgr_pixel> cimg(temp);

            // guarda los puntos obtenidos
            std::vector<image_window::overlay_circle> points;
            std::vector<full_object_detection> detects;

            // detectar los landmarks para cada rostro encontrado
            for (unsigned long i = 0; i < facesRect.size(); ++i) 
            {
                full_object_detection shape = pose_model(cimg, facesRect[i]);
                detects.push_back(shape);

                image_pts.push_back(cv::Point2d(shape.part(17).x(), shape.part(17).y())); //#17 left brow left corner
                image_pts.push_back(cv::Point2d(shape.part(21).x(), shape.part(21).y())); //#21 left brow right corner
                image_pts.push_back(cv::Point2d(shape.part(22).x(), shape.part(22).y())); //#22 right brow left corner
                image_pts.push_back(cv::Point2d(shape.part(26).x(), shape.part(26).y())); //#26 right brow right corner
                image_pts.push_back(cv::Point2d(shape.part(36).x(), shape.part(36).y())); //#36 left eye left corner
                image_pts.push_back(cv::Point2d(shape.part(39).x(), shape.part(39).y())); //#39 left eye right corner
                image_pts.push_back(cv::Point2d(shape.part(42).x(), shape.part(42).y())); //#42 right eye left corner
                image_pts.push_back(cv::Point2d(shape.part(45).x(), shape.part(45).y())); //#45 right eye right corner
                image_pts.push_back(cv::Point2d(shape.part(31).x(), shape.part(31).y())); //#31 nose left corner
                image_pts.push_back(cv::Point2d(shape.part(35).x(), shape.part(35).y())); //#35 nose right corner
                image_pts.push_back(cv::Point2d(shape.part(48).x(), shape.part(48).y())); //#48 mouth left corner
                image_pts.push_back(cv::Point2d(shape.part(54).x(), shape.part(54).y())); //#54 mouth right corner
                image_pts.push_back(cv::Point2d(shape.part(57).x(), shape.part(57).y())); //#57 mouth central bottom corner
                image_pts.push_back(cv::Point2d(shape.part(8).x(), shape.part(8).y()));   //#8 chin corner

                for(auto pt : image_pts) 
                    points.push_back(image_window::overlay_circle(dlib::point(pt.x, pt.y), 2, color));

                int max_d = MAX(temp.rows, temp.cols);

                // matriz de camara
                cv::Mat cam_matrix = (Mat_<double>(3, 3) <<
                    max_d, 0, temp.cols / 2.0,
                    0, max_d, temp.rows / 2.0,
                    0, 0, 1.0);

                // calcular pose 3D
                cv::solvePnP(object_pts, image_pts, cam_matrix, cv::noArray(), rotation_vec, translation_vec);

                // puntos 3D para generar los ejes de rotacion
                std::vector<cv::Point3d> reprojectsrc;
                reprojectsrc.push_back(cv::Point3d(0.0, 0.0, 0.0));
                reprojectsrc.push_back(cv::Point3d(15.0, 0.0, 0.0));
                reprojectsrc.push_back(cv::Point3d(0.0, 15.0, 0.0));
                reprojectsrc.push_back(cv::Point3d(0.0, 0.0, 15.0));

                // puntos 3D proyectados en 2D
                std::vector<cv::Point2d> reprojectdst;
                reprojectdst.resize(4);
        
                // reproject
                cv::projectPoints(reprojectsrc, rotation_vec, translation_vec, cam_matrix, cv::noArray(), reprojectdst);

                // dibujar los ejes de rotacion 
                line(temp, reprojectdst[0], reprojectdst[1], cv::Scalar(0, 0, 255), 2, cv::LINE_AA);
                line(temp, reprojectdst[0], reprojectdst[2], cv::Scalar(0, 255, 0), 2, cv::LINE_AA);
                line(temp, reprojectdst[0], reprojectdst[3], cv::Scalar(255, 0, 0), 2, cv::LINE_AA);

                //calc euler angle
                cv::Rodrigues(rotation_vec, rotation_mat);
                cv::hconcat(rotation_mat, translation_vec, pose_mat);
                cv::decomposeProjectionMatrix(pose_mat, out_intrinsics, out_rotation, out_translation, cv::noArray(), cv::noArray(), cv::noArray(), euler_angle);

                // fuente para texto
                int font = cv::FONT_HERSHEY_PLAIN;
                int baseline = 0;

                // dibujar pitch 
                cv::String txtPitch = cv::format("PITCH: %.2f", euler_angle.at<double>(0));
                cv::Size size = cv::getTextSize(txtPitch, font, 1.0, 1, &baseline);
                cv::Point pos = cv::Point(reprojectdst[1].x - size.width, reprojectdst[1].y);
                cv::putText(temp, txtPitch, pos, font, 1.0, cv::Scalar::all(255), 1, cv::LINE_AA);

                // dibujar yaw
                cv::String txtYaw = cv::format("YAW: %.2f", euler_angle.at<double>(1));
                cv::putText(temp, txtYaw, reprojectdst[2], font, 1.0, cv::Scalar::all(255), 1, cv::LINE_AA);

                // dibujar roll
                cv::String txtRoll = cv::format("ROLL: %.2f", euler_angle.at<double>(2));
                cv::putText(temp, txtRoll, reprojectdst[3], font, 1.0, cv::Scalar::all(255), 1, cv::LINE_AA);

                image_pts.clear();
            }

            // Display it all on the screen
            win.clear_overlay();
            win.set_image(cimg);                             // establecer la imagen en la ventana
            win.add_overlay(points);                         // dibujar los landmarks point
        }
    }
    catch (serialization_error& e) {
        cout << endl << e.what() << endl;
    }
    catch (exception& e) {
        cout << e.what() << endl;
    }
}
