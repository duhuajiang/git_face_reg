
#define CPU_ONLY 1
#define BLAS "open"

#include <vector>
#include <assert.h>
#include <math.h>
#include <iostream>
#include <string>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <sys/time.h>
#include <string>
#include <set>
#include <map>
#include <algorithm>
#include <stdio.h>
#include <sys/stat.h>
#include <dirent.h>
#include <fstream>
#include <iterator>
#include <caffe/caffe.hpp>
#include <iosfwd>
#include <memory>
#include <utility>
//#define JLH_IPU




#define  PATH_FACE_XML          "xml_path"
#define  PATH_VGG_PRO_TEXT      "prototxt"
#define  PATH_VGG_MODEL         "vgg_model"
#define  PATH_FEATURE_TXT       "featuretxt"
#define  PATH_HISTORYIMG_TXT    "historyimgtxt"
#define  PATH_IMG               "imgpath"
#define  PATH_TEST_VIDEO        "videopath"
#define  NUM_OUTPUT_FEA         "outnumber"
#define  OUT_LAYER_NAME         "outlayername"
#define  THRESHOLD              "threshold"
#define  WINDOW_NAME            "window_name"
#define  MEAN_VALUE             "meanvalue"

#define  PATH_FACE_PROTXT       "face_protxt"
#define  PATH_FACE_MODEL        "face_model"
#define  CONFIDENCE             "confidence"
#define  FACE_INWIDTH           "face_inW"
#define  FACE_INHEIGHT          "face_inH"

#define  VEDIO_WIDTH            "vedio_width"
#define  VEDIO_HEIGHT           "vedio_height"
#define  VEDIO_RATE             "vedio_rate"
#define  OUT_VEIDO_PATH         "out_vedio_path"

using namespace caffe;
using namespace std;
using namespace cv;


class Extractor {
 public:

   Extractor(const string& prototxt,const string& vgg_model,const string& mean_value);

   void SetMean(const string& mean_value);

   std::vector<float> Predict(const cv::Mat& img);

   void WrapInputLayer(std::vector<cv::Mat>* input_channels);

   void Preprocess(const cv::Mat& img,std::vector<cv::Mat>* input_channels);

  private:
      std::shared_ptr<Net<float> > net_;
      cv::Size input_geometry_;
      int num_channels_;
      cv::Mat mean_;
};

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


typedef struct  fea{
      vector<float> fea;
        string  img_lable;
}Feature;


float cosine(const vector<float>& v1, const vector<float>& v2);
void  gen_basedata(string featuretxt,string historyimgtxt,string imgpath,Extractor extractor);
map<string,vector<float> >  load_fea_name(string file_name);
const vector<string>  explode(const string& s, const char& c);
vector<float> string_to_vec(string str,char c);

//get feture from vgg

Extractor::Extractor(const string& prototxt,const string& vgg_model,const string& mean_value)
{
        #ifdef CPU_ONLY
          Caffe::set_mode(Caffe::CPU);
        #elif JLH_IPU
        ipuLibInit();
          Caffe::set_mode(Caffe::IPU);
        Caffe::set_layer_fusion(true);
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
#ifdef JLH_IPU
    net_->MallocIPUMemory();
#endif
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
      #ifdef JLH_IPU
      	ipuAddIteration();
      #endif
      Blob<float>* input_layer = net_->input_blobs()[0];
      input_layer->Reshape(1, num_channels_,input_geometry_.height, input_geometry_.width);
      net_->Reshape();

      std::vector<cv::Mat> input_channels;
      WrapInputLayer(&input_channels);

      Preprocess(img, &input_channels);


      net_->Forward(net_->input_blobs());

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


//get feature from file
vector<string>  getFiles( string path )
{
       vector<string> files;
       DIR* pDir;
       struct dirent* ptr;

       if(!(pDir = opendir(path.c_str()))){
           cout << "opendir error: " << path << endl;
           exit(0);
           }
        string subFile;
        while((ptr = readdir(pDir)) != 0){
                   subFile = ptr -> d_name;
                   if(subFile == "." || subFile == "..")
                               continue;
                   files.push_back(subFile);
               }
    closedir(pDir);

     return files;
}

const vector<string>  explode(const string& s, const char& c)
{
   string buff{""};
   vector<string> v;

   for(auto n:s)
   {
       if(n != c) buff+=n;
       else if(n == c && buff != "") { v.push_back(buff); buff = ""; }
   }
   if(buff != "") v.push_back(buff);

   return v;
}


vector<float> string_to_vec(string str,char c)
{
    vector<string> v_s = explode(str,c);
    vector<float> v_f;
    for (vector<string>::const_iterator iter = v_s.cbegin(); iter != v_s.cend(); iter++)
                   {
                       float f  = atof( (*iter).c_str());
                       v_f.push_back(f);

                   }
       return v_f;
}


vector<float> get_img_feature(string file_name,Extractor extractor)
{

       Mat img = imread(file_name);
       resize(img, img, Size(224, 224));
        vector<float> test_vector;
       if (!img.empty())
       {
           test_vector =  extractor.Predict(img);
       }else
       {
          cout<<"file "+file_name+"didn't  read"<<endl;
       }
       return test_vector;
}


map<string,vector<float> >  load_fea_name(string file_name)
{
           map<string,vector<float> > fea_map;
           ifstream fea_in(file_name, ios::in); //创建输入流对象
           if(fea_in.good())
           {
                 string  s;
               while ( getline(fea_in,s) )
               {
                   vector<string> str = explode(s,',');
                   fea_map.insert(pair<string, vector<float> >(str[1], string_to_vec(str[0],' ')));
               }
           }
           fea_in.close();
           return fea_map;

}


void gen_basedata(string featuretxt,string historyimgtxt,string imgpath,Extractor extractor)
{
       string  his_record_txt = historyimgtxt;
       string  feature_txt = featuretxt;
       string img_path = imgpath;

       set<string> his_record_set;

       ifstream fin(his_record_txt);
       if (fin)
       {
               string s;
               while(getline(fin,s))
               {
                   his_record_set.insert(s);
               }
               fin.close();
       }

       vector<string> files = getFiles(img_path);

       ofstream out(his_record_txt,ios::app|ios::out);

       ofstream  feap(feature_txt,ios::app|ios::out);

           if (out.is_open())
           {
                for (vector<string>::const_iterator iter = files.begin(); iter != files.end(); iter++)
                   {
                         string single_img_path = img_path+(*iter);
                         string  img_label = explode((*iter),'.')[0];
                         if(his_record_set.count(single_img_path)==0)
                         {
                               cout <<img_label<< ','<< single_img_path<<endl;
                               vector<float> test_vector =get_img_feature(single_img_path,extractor);

                               for (vector<float>::const_iterator f_i = test_vector.cbegin(); f_i != test_vector.cend(); f_i++)
                                   {
                                          feap<<to_string((*f_i))<<' ';
                                   }
                               feap<<','+img_label+'\n';
                               out<<single_img_path+'\n';
                         }
                   }
               feap.close();
               out.close();
           }
}




//get face by opencv

Dector::Dector(const string& face_prototxt,const string& face_model,const float confidence,const int inWidth,const int inHeight)
{
        net = dnn::readNetFromCaffe(face_prototxt, face_model);
        min_confidence = confidence;
        face_inWidth = inWidth;
        face_inHeight = inHeight;

}

Face Dector::detect_face(Mat frame)
{
    const Scalar meanVal(104.0, 177.0, 123.0);
    const double inScaleFactor = 1.0;
    if (frame.channels() == 4) cvtColor(frame, frame, COLOR_BGRA2BGR);
    Mat inputBlob = dnn::blobFromImage(frame, inScaleFactor,Size(face_inWidth, face_inHeight), meanVal, false, false);

    net.setInput(inputBlob, "data");
    Mat detection = net.forward("detection_out");
    Mat detectionMat(detection.size[2], detection.size[3], CV_32F, detection.ptr<float>());


    vector<Rect> faces;
    vector<Mat>  face_img;


    for (int i = 0; i < detectionMat.rows; i++)
        {
            float confidence = detectionMat.at<float>(i, 2);

            if (confidence > min_confidence)
            {
                int xLeftBottom = static_cast<int>(detectionMat.at<float>(i, 3) * frame.cols);
                int yLeftBottom = static_cast<int>(detectionMat.at<float>(i, 4) * frame.rows);
                int xRightTop = static_cast<int>(detectionMat.at<float>(i, 5) * frame.cols);
                int yRightTop = static_cast<int>(detectionMat.at<float>(i, 6) * frame.rows);

                Rect object((int)xLeftBottom, (int)yLeftBottom,
                        (int)(xRightTop - xLeftBottom),
                        (int)(yRightTop - yLeftBottom));

                Mat tmp = frame(object);
                face_img.push_back(tmp);
                faces.push_back(object);


            }

        }
     Face f_re={.faces = faces,.face_img = face_img};
     return  f_re;
}



//compute distance

float dotProduct(const vector<float>& v1, const vector<float>& v2)
 {
        assert(v1.size() == v2.size());
        float ret = 0.0;
        for (vector<float>::size_type i = 0; i != v1.size(); ++i)
         {
                ret += v1[i] * v2[i];
         }
        return ret;
 }

float module(const vector<float>& v)
 {
        float ret = 0.0;
        for (vector<float>::size_type i = 0; i != v.size(); ++i)
             {
                ret += v[i] * v[i];
             }
        return sqrt(ret);
}

float cosine(const vector<float>& v1, const vector<float>& v2)
{
   assert(v1.size() == v2.size());
   return dotProduct(v1, v2) / (module(v1) * module(v2));
}


//main function


void  show_vec(vector<float> v)
{
            int i =0 ;
            while (i<5)
                {
                    cout <<  v[i++] <<"   ";
                }
            cout<<".....\n";
}

void show_map(map<string,vector<float> > mp)
{
         map<string,vector<float> >::iterator iter;
         for(iter = mp.begin(); iter != mp.end(); iter++)
            {
               cout<<iter->first<<"    ";
               show_vec(iter->second);
            }
}


map<string,string> parse_parm(string  config_file)
{
      map<string,string> parm;

        ifstream fea_in(config_file, ios::in); //创建输入流对象
            if(fea_in.good())
            {
                  string  s;
                while ( getline(fea_in,s) )
                {
                    vector<string> str = explode(s,':');
                    parm.insert(pair<string, string >(str[0], str[1]));
                    cout<<str[0]<<':'<<str[1]<<endl;
                }
            }
            fea_in.close();
      return parm;
}


int main(int argc,char** argv)
{ if (argc<2)
    {
          cout<<"please   input  your   video   path!\n";
          return -1;
    }


    string config_file = argv[1];
    cout<<"config file is :"<<config_file<<endl;

    map<string,string> config_map = parse_parm(config_file);

    string prototxt = (*config_map.find(PATH_VGG_PRO_TEXT)).second;
    string vgg_model = (*config_map.find(PATH_VGG_MODEL)).second;
    string meanvalue = (*config_map.find(MEAN_VALUE)).second;
    cout<<"init face vggmodel:"<<prototxt<<" and "<<vgg_model<<endl;
    Extractor extractor(prototxt, vgg_model, meanvalue);

    string featuretxt =  (*config_map.find(PATH_FEATURE_TXT)).second;
    string historyimgtxt =  (*config_map.find(PATH_HISTORYIMG_TXT)).second;
    string imgpath = (*config_map.find(PATH_IMG)).second;
    cout<<"generate feature_map:"<<featuretxt<<" and "<<historyimgtxt<<" and "<< imgpath <<endl;
    gen_basedata(featuretxt,historyimgtxt,imgpath,extractor);

    cout<<"load  feature_map:"<< featuretxt<<endl;
    map<string,vector<float> > map_feature =  load_fea_name(featuretxt);
    cout<<"show  feature_map:"<<endl;
    show_map(map_feature);



    string window_name = (*config_map.find(WINDOW_NAME)).second;


    string vedio_path = (*config_map.find(PATH_TEST_VIDEO)).second;
    cout<<"test vedio path is :"<<vedio_path<<endl;
    float threshold = atof( (*config_map.find(THRESHOLD)).second.c_str());


    int  faceInW = atoi((*config_map.find(FACE_INWIDTH)).second.c_str());
    int  faceInH = atoi((*config_map.find(FACE_INHEIGHT)).second.c_str());
    float confidence = atof((*config_map.find(CONFIDENCE)).second.c_str());
    string face_ptxt = (*config_map.find(PATH_FACE_PROTXT)).second;
    string face_model = (*config_map.find(PATH_FACE_MODEL)).second;
    Dector dector(face_ptxt,face_model,confidence,faceInH,faceInW);



    int  vedio_width = atoi((*config_map.find(VEDIO_WIDTH)).second.c_str());
    int  vedio_height = atoi((*config_map.find(VEDIO_HEIGHT)).second.c_str());
    double vedio_rate = atof((*config_map.find(VEDIO_RATE)).second.c_str());
    string out_video_path = (*config_map.find(OUT_VEIDO_PATH)).second;

    Size videoSize(vedio_width,vedio_height);
    VideoWriter writer(out_video_path, CV_FOURCC('M', 'J', 'P', 'G'),vedio_rate, videoSize);

    VideoCapture cap(vedio_path);


         if(!cap.isOpened())
            {
                    return -1;
            }

        Mat frame;
        int flag = 0;

        cout<<"start detect:"<<endl;
        cout<<threshold<<endl;
        while(1)
        {

            struct timeval allt1, allt2;
            gettimeofday(&allt1,NULL);

            cap>>frame;
            if(frame.empty()) break;

            struct timeval dt1, dt2;
            gettimeofday(&dt1,NULL);
            Face f = dector.detect_face(frame);

            gettimeofday(&dt2,NULL);
            float delt_dt = 1000000*(dt2.tv_sec - dt1.tv_sec) + dt2.tv_usec - dt1.tv_usec;
            delt_dt /=1000000;
            cout<<"face detect:"<<delt_dt<<endl;

            int index = 0;
            for(vector<Mat>::const_iterator iter=f.face_img.begin();iter!=f.face_img.end();iter++)
            {

                        float max_p = -1.0;
                        string max_name = "unknown";
                        Mat res_mat;
                        resize(*iter, res_mat, Size(224, 224));


                        struct timeval et1, et2;
                        gettimeofday(&et1,NULL);

                        vector<float>  temp =  extractor.Predict(res_mat);

                        gettimeofday(&et2,NULL);

                        float delt_et = 1000000*(et2.tv_sec - et1.tv_sec)+ et2.tv_usec - et1.tv_usec;
                        delt_et /=1000000;
                        cout<<"feature extract:"<<delt_et<<endl;


                        for(map<string,vector<float> >::iterator mp_iter = map_feature.begin(); mp_iter != map_feature.end(); mp_iter++)
                          {
                                 float temp_p =  cosine(temp,mp_iter->second);

                                 if(temp_p>max_p)
                                 {
                                       max_p = temp_p;
                                       max_name =  mp_iter->first;
                                 }

                          }
                          if (max_p<threshold)
                          {
                              cout<<max_p<<endl;
                              max_p = 0.0;
                              max_name =  "unknown";
                          }

                          Point p = Point(f.faces[index].x,f.faces[index].y+3);
                          putText(frame, max_name+':'+to_string(max_p), p, cv::FONT_HERSHEY_TRIPLEX, 0.6, cv::Scalar(255, 200, 200), 0.2, CV_AA);
                          rectangle(frame,f.faces[index],Scalar(0,0,255),2,8); //画出脸部矩形
                          index ++;

            }

            imshow(window_name,frame);

            gettimeofday(&allt2,NULL);
            float delt_allt = 1000000*(allt2.tv_sec - allt1.tv_sec) + allt2.tv_usec - allt1.tv_usec;
            delt_allt /=1000000;
            cout<<"all process:"<<delt_allt<<endl;
            cout<<endl;
            flag++;
            writer << frame;

            if(waitKey(1) >=0)
                break;
        }

        #ifdef JLH_IPU
        ipuLibExit();
        #endif
        return 0;
}




