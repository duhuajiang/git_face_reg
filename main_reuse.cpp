#include <iostream>
#include "Communication/Commnunication_Base.h"
#include "Machine/Machine.h"
#include "Communication/User.h"
#include "Base/Data_struct.h"
#include <string>

using namespace std;


/*void Init_Monitoring_Task(task_node *node,int id){
    strcpy(node->App_name,"sss");
    sprintf(node->Output,"/home/Healthy_Detect/%d/",id);
    strcpy(node->Input,"/home/facerg/multi/invideo.txt");
}
int main(int argc,char **argv) {
    char App_name[NAME_LENGTH]="sss";
    task_node node;

    std::cout<<User_Insert_App(App_name)<<std::endl;
    if(argc!=2){
        std::cout<<"usage : Insert_Monitoring_Task output"<<std::endl;
        return 0;
    }
    int DeviceID;
    DeviceID = atoi(argv[1]);
    //std::cin>>a;
    Init_Monitoring_Task(&node,DeviceID);
    std::cout<<User_Send_Task(&node)<<std::endl;


//    std::cout<<User_Query_Ap0p_Live(App_name,DeviceID)<<std::endl;
//    std::cout<<User_Live_App(App_name,DeviceID)<<std::endl;
//    std::cout<<User_Query_App_Live(App_name,DeviceID)<<std::endl;
    // std::cout<<User_Delete_App(App_name,DeviceID)<<std::endl;
//    std::cout<<User_Query_App_Live(App_name,DeviceID)<<std::endl;
    return 0;
}
*/

int main(int argc,char **argv) {
    std::string App_name = argv[1];
    std::string input = argv[2];
    int DeviceID = atoi(argv[4]);
    std::string output = argv[3]+to_string(DeviceID)+'/';

    task_node node;
    strcpy(node.App_name,App_name.c_str());
    strcpy(node.Output,output.c_str());
    strcpy(node.Input,input.c_str());


    std::cout<<User_Insert_App(node.App_name)<<std::endl;
    if(argc!=5){
        std::cout<<"usage : Insert_Monitoring_Task output"<<std::endl;
        return 0;
    }
    std::cout<<User_Send_Task(&node)<<std::endl;

//    std::cout<<User_Query_Ap0p_Live(App_name,DeviceID)<<std::endl;
//    std::cout<<User_Live_App(App_name,DeviceID)<<std::endl;
//    std::cout<<User_Query_App_Live(App_name,DeviceID)<<std::endl;
    // std::cout<<User_Delete_App(App_name,DeviceID)<<std::endl;
//    std::cout<<User_Query_App_Live(App_name,DeviceID)<<std::endl;
    return 0;
}
