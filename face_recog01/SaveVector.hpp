#ifndef SAVEVECTOR_HPP_INCLUDED
#define SAVEVECTOR_HPP_INCLUDED


#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/gpu/gpu.hpp>
#include <fstream>
#include <sstream>
#include <iostream>
using namespace cv;
using namespace std;

void SaveNameVector(vector<string>   &NameVector, string filename);//保存姓名，需要输入
vector<string> LoadNameVector(vector<string>   &NameVector, string filename);
void SaveFaceMatrix(float **FaceMatrix, string filename, int rows);//用于保存提取出来特征的人脸矩阵，需要输入：人脸矩阵、保存的文件名，矩阵的行数（列均为2622维）
Mat LoadMat(string file);//将xml文件提取出来转换为OpenCV的Mat类
float** MatToVector2d(Mat &FaceMatrix_mat);//将Mat类转换为二维数组

#endif // SAVEVECTOR_HPP_INCLUDED
