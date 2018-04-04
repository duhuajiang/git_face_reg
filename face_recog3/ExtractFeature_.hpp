#ifndef EXTRACTFEATURE__HPP_INCLUDED
#define EXTRACTFEATURE__HPP_INCLUDED

#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
using namespace cv;
using namespace std;
std::vector<float> ExtractFeature(Mat FaceROI,string layer_name,int out_vec_num);//给一个图片 返回一个vector<float>容器
void Caffe_Predefine(std::string param_file, std::string pretrained_param_file);
float* ExtractFeature_(Mat FaceROI,string layer_name,int out_vec_num);
vector<float> ExtractFeature1(Mat FaceROI,string layer_name,int out_vec_num);
#endif // EXTRACTFEATURE__HPP_INCLUDED
