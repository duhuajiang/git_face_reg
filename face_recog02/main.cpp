#include <iostream>
#include "Dect_face.hpp"
//#include "ExtractFeature_.hpp"
#include "MyExtractFeature.hpp"
#include "ComputeDistance.hpp"
#include "Gen_BaseData.hpp"
#include <time.h>

using namespace std;
using namespace cv;





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
{
        if (argc<2)
        {
              cout<<"please   input  your   video   path!\n";
              return -1;
        }


        string config_file = argv[1];
        cout<<"config file is :"<<config_file<<endl;

        map<string,string> config_map = parse_parm(config_file);

        string xmlpath = (*config_map.find(PATH_FACE_XML)).second;

        string prototxt = (*config_map.find(PATH_VGG_PRO_TEXT)).second;

        string vgg_model = (*config_map.find(PATH_VGG_MODEL)).second;

        string featuretxt =  (*config_map.find(PATH_FEATURE_TXT)).second;

        string historyimgtxt =  (*config_map.find(PATH_HISTORYIMG_TXT)).second;

        string imgpath = (*config_map.find(PATH_IMG)).second;

        string meanvalue = (*config_map.find(MEAN_VALUE)).second;

        string window_name = (*config_map.find(WINDOW_NAME)).second;

        string vedio_path = (*config_map.find(PATH_TEST_VIDEO)).second;

        cout<<"init face dect model:"<<xmlpath<<endl;
        CascadeClassifier ccf ;
        ccf.load(xmlpath);

        cout<<"init face vggmodel:"<<prototxt<<" and "<<vgg_model<<endl;
        Extractor extractor(prototxt, vgg_model, meanvalue);

        cout<<"generate feature_map:"<<featuretxt<<" and "<<historyimgtxt<<" and "<< imgpath <<endl;
        gen_basedata(featuretxt,historyimgtxt,imgpath,extractor);

        cout<<"load  feature_map:"<< featuretxt<<endl;
        map<string,vector<float> > map_feature =  load_fea_name(featuretxt);

        cout<<"show  feature_map:"<<endl;
        show_map(map_feature);
        cout<<"test vedio path is :"<<vedio_path<<endl;


        float threshold = atof( (*config_map.find(THRESHOLD)).second.c_str());

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


            cap>>frame;
            if(frame.empty()) break;

            Face f = detect_face( frame,ccf);

            int index = 0;
            for(vector<Mat>::const_iterator iter=f.face_img.begin();iter!=f.face_img.end();iter++)
            {

                        float max_p = -1.0;
                        string max_name = "unknown";
                        Mat res_mat;
                        resize(*iter, res_mat, Size(224, 224));


                        clock_t start = clock();

                        vector<float>  temp =  extractor.Predict(res_mat);

                        printf( "%f seconds\n", (double)( clock() - start) / CLOCKS_PER_SEC );

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
            flag++;

            if(waitKey(1) >=0)
                break;
        }

        return 0;
}
