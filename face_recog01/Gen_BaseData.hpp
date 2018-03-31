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
#include "ExtractFeature_.hpp"
#include <iterator>


typedef struct  fea{
      vector<float> fea;
        string  img_lable;
}Feature;

using namespace std;
void  gen_basedata(string  base_root_file);
map<string,vector<float> >  load_fea_name(string file_name);
const vector<string>  explode(const string& s, const char& c);
#endif // GEN_BASEDATA_HPP_INCLUDED
