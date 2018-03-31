#include "ExtractFeature_.hpp"
#include "Caffe_net_memorylayer.hpp"
using namespace caffe;

/*namespace caffe
{
            extern INSTANTIATE_CLASS(InputLayer);
            extern INSTANTIATE_CLASS(InnerProductLayer);
            extern INSTANTIATE_CLASS(DropoutLayer);
            extern INSTANTIATE_CLASS(ConvolutionLayer);
            REGISTER_LAYER_CLASS(Convolution);
            extern INSTANTIATE_CLASS(ReLULayer);
            REGISTER_LAYER_CLASS(ReLU);
            extern INSTANTIATE_CLASS(PoolingLayer);
            REGISTER_LAYER_CLASS(Pooling);
            extern INSTANTIATE_CLASS(LRNLayer);
            REGISTER_LAYER_CLASS(LRN);
            extern INSTANTIATE_CLASS(SoftmaxLayer);
            REGISTER_LAYER_CLASS(Softmax);
            extern INSTANTIATE_CLASS(MemoryDataLayer);
}*/

template <typename Dtype> caffe::Net<Dtype>* Net_Init_Load(std::string param_file, std::string pretrained_param_file, caffe::Phase phase)
{
            caffe::Net<Dtype>* net(new caffe::Net<Dtype>(param_file, caffe::TEST));
            net->CopyTrainedLayersFrom(pretrained_param_file);
            return net;
}

void Caffe_Predefine(std::string param_file, std::string pretrained_param_file)//when our code begining run must add it
{
            caffe::Caffe::set_mode(caffe::Caffe::CPU);
            net = Net_Init_Load<float>(param_file, pretrained_param_file, caffe::TEST);
            memory_layer = (caffe::MemoryDataLayer<float> *)net->layers()[0].get();
}

std::vector<float> ExtractFeature(Mat FaceROI)
{
            caffe::Caffe::set_mode(caffe::Caffe::CPU);
            std::vector<Mat> test;
            std::vector<int> testLabel;
            std::vector<float> test_vector;
            test.push_back(FaceROI);
            testLabel.push_back(0);
            memory_layer->AddMatVector(test, testLabel);// memory_layer and net , must be define be a global variable.
            test.clear(); testLabel.clear();
            std::vector<caffe::Blob<float>*> input_vec;
            net->Forward(input_vec);
            boost::shared_ptr<caffe::Blob<float>> fc8 = net->blob_by_name("fc8");
            int test_num = 0;
            while (test_num < 2622)
            {
                test_vector.push_back(fc8->data_at(0, test_num++, 1, 1));
            }
            return test_vector;
}
