#include <iostream>
#include "Dect_face.hpp"
#include "ExtractFeature_.hpp"
#include "ComputeDistance.hpp"
#include "Gen_BaseData.hpp"
#include <time.h>

using namespace std;
using namespace cv;
#define  XML_PATH    "xml_path"
#define  PRO_TEXT    "pro_text"
#define  VGG_MODEL  "vgg_model"
#define  BASE_DATA  "base_data"
#define  VIDEO_PATH "video_path"

CascadeClassifier ccf ;
map<string,vector<float> > map_f ;



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

int init(map<string,string> config)
{

      string xmlpath = (*config.find(XML_PATH)).second;
      string protxt = (*config.find(PRO_TEXT)).second;
      string model = (*config.find(VGG_MODEL)).second;
      string basedata = (*config.find(BASE_DATA)).second;



    cout<<"init face dect model:"<<xmlpath<<endl;
    ccf.load(xmlpath);

    cout<<"init face recongnition:"<<protxt<<" & "<<model<<endl;
    Caffe_Predefine(protxt, model);

    cout<<"generate  feature_map:"<<basedata<<endl;
    gen_basedata(basedata);
    string  fea_path = basedata+"fea.txt";
    cout<<"load  feature_map:"<< fea_path<<endl;
    map_f =  load_fea_name(fea_path);
    cout<<"show  feature_map:\n";
     show_map(map_f);
    return 0;
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
{
        if (argc<2)
        {
              cout<<"please   input  your   video   path!\n";
              return -1;
        }
        string config_file = argv[1];
        cout<<config_file<<endl;
        map<string,string> config_map = parse_parm(config_file);
       init(config_map);

      string window_name = "face recognition";
     string vedio_path = (*config_map.find(VIDEO_PATH)).second;
     cout<<vedio_path<<endl;
     VideoCapture cap(vedio_path);

     if(!cap.isOpened())
        {
                return -1;
        }

        Mat frame;
        int flag = 0;
        cout<<"start detect:"<<endl;
        while(1)
        {


            cap>>frame;
            if(frame.empty()) break;

           /* if (flag%5!=0)
            {
                flag++;
                 continue;
            }*/

            Face f = detect_face( frame,ccf);


          /* for(vector<Rect>::const_iterator iter=f.faces.begin();iter!=f.faces.end();iter++)
                {
                    rectangle(frame,*iter,Scalar(0,0,255),2,8); //画出脸部矩形
                }

*/
            int index = 0;
            for(vector<Mat>::const_iterator iter=f.face_img.begin();iter!=f.face_img.end();iter++)
            {

                        float max_p = -1.0;
                        string max_name = "unknown";
                        Mat res_mat;
                         resize(*iter, res_mat, Size(224, 224));
                            long    i = 10000000L;
                            clock_t start, finish;
                            double  duration=0.0;
                            start = clock();
                        vector<float>  temp =  ExtractFeature(res_mat);
                            finish = clock();
                            duration = (double)(finish - start) / CLOCKS_PER_SEC;
                            printf( "%f seconds\n", duration );

                       for(map<string,vector<float> >::iterator mp_iter = map_f.begin(); mp_iter != map_f.end(); mp_iter++)
                          {
                                 float temp_p =  cosine(temp,mp_iter->second);
                                 if(temp_p>max_p)
                                 {
                                       max_p = temp_p;
                                       max_name =  mp_iter->first;
                                 }

                          }
                          if (max_p<0.75)
                          {
                              max_p = 0.0;
                              max_name =  "unknown";
                          }

                          Point p = Point(f.faces[index].x,f.faces[index].y+3);
                          putText(frame, max_name+':'+to_string(max_p), p, cv::FONT_HERSHEY_TRIPLEX, 0.6, cv::Scalar(255, 200, 200), 0.2, CV_AA);
                          rectangle(frame,f.faces[index],Scalar(0,0,255),2,8); //画出脸部矩形
                          index ++;

            }

            imshow(window_name,frame);
            flag++;

            if(waitKey(1) >=0)
                break;
        }

        return 0;
}
