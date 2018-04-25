#include <vector>
#include <assert.h>
#include <sys/types.h>
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
#include "Communication.h"

//#define JLH_IPU


#define  PATH_VGG_PRO_TEXT      "prototxt"
#define  PATH_VGG_MODEL         "vgg_model"
#define  MEAN_VALUE             "meanvalue"

#define  PATH_FEATURE_TXT       "featuretxt"
#define  PATH_HISTORYIMG_TXT    "historyimgtxt"

#define  PATH_IMG               "imgpath"
#define  THRESHOLD              "threshold"

#define  PATH_FACE_PROTXT       "face_protxt"
#define  PATH_FACE_MODEL        "face_model"
#define  CONFIDENCE             "confidence"
#define  FACE_INWIDTH           "face_inW"
#define  FACE_INHEIGHT          "face_inH"
#define  REMOVE_HISTORY         "removehistroy"

#define  IMG_W                  224
#define  IMG_H                  224


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

typedef struct res_face_rate{
      cv::Mat face;
      float   maxRate;
}Res;

typedef struct v{

    float  sum_time;
    float  process_time;
    string video_name;
    string video_path;
    int sum_frame;
    int sum_face;
    //string person;
    std::map<string,Res> result;

}Video_info;

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
      Blob<float>* input_layer = net_->input_blobs()[0];
      num_channels_ = input_layer->channels();
      CHECK(num_channels_ == 3 || num_channels_ == 1)<< "Input layer should have 1 or 3 channels.";
      input_geometry_ = cv::Size(input_layer->width(), input_layer->height());
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
       resize(img, img, Size(IMG_W, IMG_H));
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

                try
                {
                    Mat tmp = frame(object).clone();
                    face_img.push_back(tmp.clone());
                    faces.push_back(object);
                }
                catch (exception& e)
                {
                    continue;
                }


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

vector<string> get_video_path(string file_path)
{
    vector<string> file_list;

    ifstream fin(file_path);
    if (fin)
    {
            string s;
            while(getline(fin,s))
            {
                file_list.push_back(s);
            }
            fin.close();
    }
    return file_list;

}

Video_info process_single_video(string in_video,string out_dir,Dector dector,Extractor extractor,float threshold, map<string,vector<float> > map_feature)
{

    set<string> name_list;

    struct timeval allt1, allt2;
    gettimeofday(&allt1,NULL);

    vector<string> path_v = explode(in_video,'/');

    vector<string> video_name = explode((path_v[path_v.size()-1]),'.');
    string out_path = out_dir + video_name[0] + ".avi";


    VideoCapture cap(in_video);

    Size vedio_size((int)cap.get(CV_CAP_PROP_FRAME_WIDTH ),(int)cap.get(CV_CAP_PROP_FRAME_HEIGHT ));
    double vedio_rate = cap.get(CV_CAP_PROP_FPS);
    int vedio_frame_count = (int)cap.get(CV_CAP_PROP_FRAME_COUNT);
    float video_time = (float)vedio_frame_count/vedio_rate;
    VideoWriter writer(out_path, CV_FOURCC('X', 'V', 'I', 'D'),vedio_rate, vedio_size,1);



    if(!cap.isOpened())
       {
          exit(1);
       }


     int sum_frame = 0;
     int sum_face = 0;
     Video_info  v_f;
        while(1)
        {

            Mat frame;
            cap>>frame;
            if(frame.empty()) break;

            Face f;

            try{
                  f = dector.detect_face(frame);
            }catch (exception& e)
            {
                continue;
            }




            int index_frame = 0;
            int unkown_face_num = 0;
            for(vector<Mat>::const_iterator iter=f.face_img.begin();iter!=f.face_img.end();iter++)
            {

                    float max_p = -1.0;
                    string max_name = "unknown";

                    Mat res_mat;
                    resize(*iter, res_mat, Size(IMG_W, IMG_H));



                     vector<float>  temp;

                     try{
                             temp =  extractor.Predict(res_mat);
                         }catch (exception& e)
                           {
                                continue;
                           }


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
//                              cout<<max_p<all<endl;
                              max_p = 0.0;
                              max_name =  "unknown";
                              unkown_face_num+=1;
                          }
                       Res tempPerson;
                       tempPerson.maxRate=max_p;
                       tempPerson.face=res_mat;

                       map<string,Res>::iterator faceiter=v_f.result.find(max_name);
                       if(faceiter==v_f.result.end()){
                          v_f.result.insert(pair<string,Res>(max_name,tempPerson));
                       }else if(faceiter->second.maxRate<max_p){
                          v_f.result[max_name]=tempPerson;
                       }

                       Point p = Point(f.faces[index_frame].x,f.faces[index_frame].y+3);
                       putText(frame, max_name+':'+to_string(max_p).substr(0,4), p, cv::FONT_HERSHEY_TRIPLEX, 0.6, cv::Scalar(255, 200, 200), 0.2, CV_AA);
                       rectangle(frame,f.faces[index_frame],Scalar(0,0,255),2,8); //画出脸部矩形
                       index_frame ++;
                       if(name_list.count(max_name)==0 && max_p!=0.0)
                       {
                           name_list.insert(max_name);
                       }


            }

            sum_face += index_frame;

            int face_num = f.faces.size();
            int know_num = face_num - unkown_face_num;
            putText(frame, "face_num:"+to_string(face_num),Point(20,45), cv::FONT_HERSHEY_TRIPLEX, 0.4, cv::Scalar(0,255,127), 0.1, CV_AA);
            putText(frame, "known_num:"+to_string(know_num), Point(20,70), cv::FONT_HERSHEY_TRIPLEX, 0.4, cv::Scalar(0,255,127), 0.1, CV_AA);

            //imshow("ddd",frame);
            writer << frame;
            sum_frame++;
            if(waitKey(1) >=0)
                break;
        }
     gettimeofday(&allt2,NULL);
     float delt_allt = 1000000*(allt2.tv_sec - allt1.tv_sec) + allt2.tv_usec - allt1.tv_usec;
     delt_allt /=1000000;
     //     cout<<"all process:"<<delt_allt<<endl;
     v_f.process_time = delt_allt;
     v_f.video_name = video_name[0];
     v_f.video_path = out_path;
     v_f.sum_frame = sum_frame;
     v_f.sum_face = sum_face;
     v_f.sum_time = video_time;
     /*for(map<string,Res>::iterator iter = v_f.result.begin();iter!= v_f.result.end();iter++){
          string imageName=iter->first+".jpg";
          imwrite(imageName,iter->second.face);
        }
        */
     //for(set<string>::iterator iter=name_list.begin();iter!=name_list.end();++iter)
        //{
           //v_f.person+=(*iter)+' ';
        //}

     return v_f;
}

int main(int argc,char** argv)
{
    if (argc<2)
    {
          cout<<"please   input  your   config   path!\n";
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

    int remove_history = atoi((*config_map.find(REMOVE_HISTORY)).second.c_str());

    if(remove_history==1)
    {
      remove(featuretxt.c_str());
      remove(historyimgtxt.c_str());

    }
    string imgpath = (*config_map.find(PATH_IMG)).second;
    cout<<"generate feature_map:"<<featuretxt<<" and "<<historyimgtxt<<" and "<< imgpath <<endl;
    gen_basedata(featuretxt,historyimgtxt,imgpath,extractor);

    cout<<"load  feature_map:"<< featuretxt<<endl;
    map<string,vector<float> > map_feature =  load_fea_name(featuretxt);
    cout<<"show  feature_map:"<<endl;
    show_map(map_feature);

    float threshold = atof( (*config_map.find(THRESHOLD)).second.c_str());
    cout<<"threshold:"<<threshold<<endl;

    int  faceInW = atoi((*config_map.find(FACE_INWIDTH)).second.c_str());
    int  faceInH = atoi((*config_map.find(FACE_INHEIGHT)).second.c_str());
    float confidence = atof((*config_map.find(CONFIDENCE)).second.c_str());
    string face_ptxt = (*config_map.find(PATH_FACE_PROTXT)).second;
    string face_model = (*config_map.find(PATH_FACE_MODEL)).second;
    Dector dector(face_ptxt,face_model,confidence,faceInH,faceInW);





    const int DeviceID = atoi(argv[2]);
    int connect_fd;
    connect_fd=init_caffe_client(DeviceID);
    int port ;

    if((recv(connect_fd, &port, sizeof(int), 0) == -1)){
        perror("recv error");
        close(connect_fd);
        return 0;
    }
    int socketfd = init_caffe_server(port);
    Caffe_reply(connect_fd);
    std::cout<<"JLH1"<<std::endl;
    close(connect_fd);

    std::cout<<"JLH3"<<std::endl;

    while(1){
        std::string file;
        char output_file_name[NAME_LENGTH];
        char Input_file[NAME_LENGTH];
        std::cout<<"JLH2"<<std::endl;
        int operation_code=Caffe_receive(Input_file,output_file_name,socketfd,&connect_fd);
        std::cout<<"Input_path"<<Input_file<<std::endl;
                std::cout<<"Output_path"<<output_file_name<<std::endl;
        if(operation_code != CAFFE_TASK){
            std::cout<<"Finish this task"<<std::endl;
            Caffe_reply(connect_fd);
            close(connect_fd);
            #ifdef JLH_IPU
            ipuLibExit();
            #endif
            return 0;
        }

        string out_dir = output_file_name;
        if (access(output_file_name, F_OK) < 0)
        {
                   if (mkdir(output_file_name, 0755) < 0)
                   {
                               cout<<"can not create path:"<<output_file_name<<endl;
                               exit(1);
                    }
        }

        string head_img_dir = out_dir+"img/";
        if (access(head_img_dir.c_str(), F_OK) < 0)
        {
                   if (mkdir(head_img_dir.c_str(), 0755) < 0)
                   {
                               cout<<"can not create path:"<<head_img_dir.c_str()<<endl;
                               exit(1);
                    }
        }

        vector<string> in_list = get_video_path(Input_file);

        for(vector<string>::const_iterator it_in_f=in_list.begin();it_in_f!=in_list.end();it_in_f++)
        {

            Video_info vinfo = process_single_video(*it_in_f,out_dir,dector,extractor,threshold,map_feature);

            cout<<"{"<<endl;
            cout<<"path_out_video:"<<vinfo.video_path<<endl;
            cout<<"all_process_time:"<<vinfo.process_time<<endl;
            cout<<"video_time:"<<vinfo.sum_time<<endl;
            cout<<"sum_frame:"<<vinfo.sum_frame<<endl;
            cout<<"sum_face:"<<vinfo.sum_face<<endl;
            //cout<<"person:"<<vinfo.person<<endl;

            string person = "[ ";
            for(map<string,Res>::iterator iter = vinfo.result.begin();iter!= vinfo.result.end();iter++){
                string imageName=head_img_dir + vinfo.video_name+'_'+iter->first+".jpg";
                imwrite(imageName,iter->second.face);
                person += "{name:"+iter->first+", max_p:"+to_string(iter->second.maxRate).substr(0,4)+",head_img_path:"+imageName+"},";
            }
             person = person.substr(0, person.size()-1) + ']';
            //person = person[person.size()-1] + ']';
            cout<<"person:"<<person<<endl;
            cout<<"}"<<endl;
            cout<<endl;


        }

        Caffe_reply(connect_fd);
        close(connect_fd);
        std::cout<<"Finish Task"<<std::endl;

    }




    cout<<"TASKCOMPLETE"<<endl;
        #ifdef JLH_IPU
        ipuLibExit();
        #endif
        return 0;
}




