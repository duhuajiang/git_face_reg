#ifndef READ_VIDEO_HPP_INCLUDED
#define READ_VIDEO_HPP_INCLUDED

#include <iostream>
#include <string>
#include <opencv2/dnn.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
using namespace cv;
using namespace std;



typedef struct  f{
      vector<Rect> faces;
       vector<Mat>  face_img;
}Face;

class Dector {
 public:

   Dector(const string& face_prototxt,const string& face_model,const float confidence,const int inWidth,const int inHeight);
   Face detect_face(Mat frame);

 private:
    dnn::Net net;
    float min_confidence;
    int face_inWidth;
    int face_inHeight;
};


#endif // READ_VIDEO_HPP_INCLUDED
