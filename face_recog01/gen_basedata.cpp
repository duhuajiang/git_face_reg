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


vector<float> string_to_vec(string str)
{
     vector<string> v_s = explode(str,' ');
     vector<float> v_f;
     for (vector<string>::const_iterator iter = v_s.cbegin(); iter != v_s.cend(); iter++)
                    {
                        float f  = atof( (*iter).c_str());
                        v_f.push_back(f);

                    }
        return v_f;
}


vector<float> get_img_feature(string file_name)
{

        Mat img = imread(file_name);
        resize(img, img, Size(224, 224));
         vector<float> test_vector;
        if (!img.empty())
        {
            test_vector =  ExtractFeature(img);
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
                    fea_map.insert(pair<string, vector<float> >(str[1], string_to_vec(str[0])));
                }
            }
            fea_in.close();
            return fea_map;

}



void write_fea_to_file(ofstream out_fea, vector<float> v_f, string lable)
{

}

void gen_basedata(string  base_root_file)
{
        string  record_file = base_root_file+"out.txt";
        string  feature_file = base_root_file + "fea.txt";
        string img_file = base_root_file+"img/";

        set<string> record_set;

        ifstream fin(record_file);
        if (fin)
        {
                string s;
                while(getline(fin,s))
                {
                    record_set.insert(s);
                }
                fin.close();
        }

          /*//遍历set
                  set<string>::iterator s_iter=record_set.begin();

                    while(s_iter!=record_set.end())
                    {
                        cout<<*s_iter<<endl;
                        ++s_iter;
                    }
            */
          vector<string> files = getFiles(img_file);

        /* //遍历vector
                for (vector<string>::const_iterator iter = files.cbegin(); iter != files.cend(); iter++)
                    {
                        cout << (*iter) << endl;
                    }
            */

            ofstream out(record_file,ios::app|ios::out);
          //  fid =fopen('D:\\\workspa\\cpp\\fileIOTest\\dataIn.txt','wt');
            ofstream  feap(feature_file,ios::app|ios::out);
            if (out.is_open())
            {
                 for (vector<string>::const_iterator iter = files.cbegin(); iter != files.cend(); iter++)
                    {
                          string img_path = img_file+(*iter);
                          string  img_label = explode((*iter),'.')[0];
                          if(record_set.count(img_path)==0)
                          {
                                cout <<img_label<< ','<< img_path<<endl;
                                vector<float> test_vector =get_img_feature(img_path);

                                for (vector<float>::const_iterator f_i = test_vector.cbegin(); f_i != test_vector.cend(); f_i++)
                                    {
                                           feap<<to_string((*f_i))<<' ';
                                    }
                                    feap<<','+img_label+'\n';
                                /*Feature tf;
                                tf.fea = test_vector;
                                tf.img_lable = img_label;
                                feap.write(reinterpret_cast<char *>(&tf), sizeof(Feature));
                                */
                                out<<img_path+'\n';
                          }
                    }
                feap.close();
                out.close();
            }
}

