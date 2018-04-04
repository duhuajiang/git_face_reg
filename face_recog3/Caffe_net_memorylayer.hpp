#ifndef CAFFE_NET_MEMORYLAYER_HPP_INCLUDED
#define CAFFE_NET_MEMORYLAYER_HPP_INCLUDED

#include <iostream>
#include <caffe/caffe.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <caffe/layers/memory_data_layer.hpp>

caffe::MemoryDataLayer<float> *memory_layer;
caffe::Net<float>* net;


#endif // CAFFE_NET_MEMORYLAYER_HPP_INCLUDED
