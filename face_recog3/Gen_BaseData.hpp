#ifndef GEN_BASEDATA_HPP_INCLUDED
#define GEN_BASEDATA_HPP_INCLUDED

#include <string>
#include <vector>
#include <set>
#include <map>
#include <iostream>
#include <algorithm>
#include <stdio.h>
#include <sys/stat.h>
#include <dirent.h>
#include <fstream>
#include "MyExtractFeature.hpp"
#include <iterator>

using namespace std;
using namespace cv;
typedef struct  fea{
      vector<float> fea;
        string  img_lable;
}Feature;


void  gen_basedata(string featuretxt,string historyimgtxt,string imgpath,Extractor extractor);

map<string,vector<float> >  load_fea_name(string file_name);
const vector<string>  explode(const string& s, const char& c);
vector<float> string_to_vec(string str,char c);
#endif // GEN_BASEDATA_HPP_INCLUDED
