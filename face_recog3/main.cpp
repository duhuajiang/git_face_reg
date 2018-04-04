#include <iostream>
#include "Dect_face.hpp"
#include "MyExtractFeature.hpp"
#include "ComputeDistance.hpp"
#include "Gen_BaseData.hpp"
#include <sys/time.h>

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

#define  PATH_FACE_PROTXT       "face_protxt"
#define  PATH_FACE_MODEL        "face_model"
#define  CONFIDENCE             "confidence"
#define  FACE_INWIDTH           "face_inW"
#define  FACE_INHEIGHT          "face_inH"

#define  VEDIO_WIDTH            "vedio_width"
#define  VEDIO_HEIGHT           "vedio_height"
#define  VEDIO_RATE             "vedio_rate"
#define  OUT_VEIDO_PATH         "out_vedio_path"


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
            }else{
               cout<<"open config file faild!"<<endl;
               exit(0);
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

        string prototxt = (*config_map.find(PATH_VGG_PRO_TEXT)).second;
        cout<<prototxt;
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


                        clock_t start = clock();

                        struct timeval et1, et2;
                        gettimeofday(&et1,NULL);

                        vector<float>  temp =  extractor.Predict(res_mat);

                        gettimeofday(&et2,NULL);
                        float delt_et = 1000000*(et2.tv_sec - et1.tv_sec)+ et2.tv_usec - et1.tv_usec;
                        delt_et /=1000000;
                        cout<<"feature extract:"<<delt_et<<endl;

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
            gettimeofday(&allt2,NULL);
            float delt_allt = 1000000*(allt2.tv_sec - allt1.tv_sec) + allt2.tv_usec - allt1.tv_usec;
            delt_allt /=1000000;
            cout<<"all process:"<<delt_allt<<endl;
            cout<<endl;
            //writer << frame;


            if(waitKey(1) >=0)
                break;
        }

        return 0;
}
