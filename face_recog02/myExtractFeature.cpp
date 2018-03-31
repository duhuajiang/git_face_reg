#include "MyExtractFeature.hpp"
#include "Gen_BaseData.hpp"

using namespace caffe;
using namespace std;


Extractor::Extractor(const string& prototxt,const string& vgg_model,const string& mean_value)
{
        #ifdef CPU_ONLY
          Caffe::set_mode(Caffe::CPU);
        #else
          Caffe::set_mode(Caffe::GPU);
        #endif


      net_.reset(new Net<float>(prototxt, TEST));
      net_->CopyTrainedLayersFrom(vgg_model);

      //CHECK_EQ(net_->num_inputs(), 1) << "Network should have exactly one input.";
      //CHECK_EQ(net_->num_outputs(), 1) << "Network should have exactly one output.";

      Blob<float>* input_layer = net_->input_blobs()[0];

      num_channels_ = input_layer->channels();
      CHECK(num_channels_ == 3 || num_channels_ == 1)<< "Input layer should have 1 or 3 channels.";

      input_geometry_ = cv::Size(input_layer->width(), input_layer->height());

      /* Load the binaryproto mean file. */
      SetMean(mean_value);

}

void Extractor::SetMean(const string& mean_value) {
       vector<float> values = string_to_vec(mean_value,' ');

       std::vector<cv::Mat> channels;
       for (int i = 0; i < num_channels_; ++i) {
      /* Extract an individual channel. */
      cv::Mat channel(input_geometry_.height, input_geometry_.width, CV_32FC1,cv::Scalar(values[i]));
      channels.push_back(channel);
    }
    cv::merge(channels, mean_);
}


std::vector<float> Extractor::Predict(const cv::Mat& img) {

      Blob<float>* input_layer = net_->input_blobs()[0];
      input_layer->Reshape(1, num_channels_,input_geometry_.height, input_geometry_.width);
      net_->Reshape();

      std::vector<cv::Mat> input_channels;
      WrapInputLayer(&input_channels);

      Preprocess(img, &input_channels);


      net_->Forward();

      Blob<float>* output_layer = net_->output_blobs()[0];

      const float* begin = output_layer->cpu_data();
      const float* end = begin + output_layer->channels();
      return std::vector<float>(begin, end);

}


void Extractor::WrapInputLayer(std::vector<cv::Mat>* input_channels) {
  Blob<float>* input_layer = net_->input_blobs()[0];

  int width = input_layer->width();
  int height = input_layer->height();
  float* input_data = input_layer->mutable_cpu_data();
  for (int i = 0; i < input_layer->channels(); ++i) {
    cv::Mat channel(height, width, CV_32FC1, input_data);
    input_channels->push_back(channel);
    input_data += width * height;
  }
}


void Extractor::Preprocess(const cv::Mat& img,std::vector<cv::Mat>* input_channels) {
  /* Convert the input image to the input image format of the network. */
  cv::Mat sample;
  if (img.channels() == 3 && num_channels_ == 1)
    cv::cvtColor(img, sample, cv::COLOR_BGR2GRAY);
  else if (img.channels() == 4 && num_channels_ == 1)
    cv::cvtColor(img, sample, cv::COLOR_BGRA2GRAY);
  else if (img.channels() == 4 && num_channels_ == 3)
    cv::cvtColor(img, sample, cv::COLOR_BGRA2BGR);
  else if (img.channels() == 1 && num_channels_ == 3)
    cv::cvtColor(img, sample, cv::COLOR_GRAY2BGR);
  else
    sample = img;

  cv::Mat sample_resized;
  if (sample.size() != input_geometry_)
    cv::resize(sample, sample_resized, input_geometry_);
  else
    sample_resized = sample;
  cv::Mat sample_float;
  if (num_channels_ == 3)
    sample_resized.convertTo(sample_float, CV_32FC3);
  else
    sample_resized.convertTo(sample_float, CV_32FC1);


  cv::Mat sample_normalized;
  cv::subtract(sample_float, mean_, sample_normalized);
  cv::split(sample_normalized, *input_channels);
  //cv::split(sample_float, *input_channels);


}

