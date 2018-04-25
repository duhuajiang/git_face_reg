#include <iostream>
#include "Communication/Commnunication_Base.h"
#include "Machine/Machine.h"
#include "Communication/User.h"
#include "Base/Data_struct.h"

int main(int argc,char **argv) {
   Machine *hui;
   hui = new Machine(3);
   hui->Start();
   delete hui;
   return 0;
}
