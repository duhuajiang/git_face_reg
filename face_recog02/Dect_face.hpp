#ifndef READ_VIDEO_HPP_INCLUDED
#define READ_VIDEO_HPP_INCLUDED

#include <iostream>
#include <string>
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
Face detect_face(Mat image,CascadeClassifier ccf);

#endif // READ_VIDEO_HPP_INCLUDED
