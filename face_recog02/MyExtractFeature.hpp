#ifndef MYEXTRACTFEATURE_HPP_INCLUDED
#define MYEXTRACTFEATURE_HPP_INCLUDED

#include <caffe/caffe.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <algorithm>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>


using namespace caffe;
using namespace std;


class Extractor {
 public:

   Extractor(const string& prototxt,const string& vgg_model,const string& mean_value);

   void SetMean(const string& mean_value);

  std::vector<float> Predict(const cv::Mat& img);

  void WrapInputLayer(std::vector<cv::Mat>* input_channels);

  void Preprocess(const cv::Mat& img,std::vector<cv::Mat>* input_channels);

 private:
  boost::shared_ptr<Net<float> > net_;
  cv::Size input_geometry_;
  int num_channels_;
  cv::Mat mean_;
};

#endif
