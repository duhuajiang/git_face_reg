#include "Gen_BaseData.hpp"


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

