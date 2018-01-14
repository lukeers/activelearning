#include "model.h"
#include <iostream>
#include <string>
#include <cstdlib>
using namespace std;
#include <ctime>
bool syn = false;

bool liveTraining = false;
string imageLoc = "/home/npillai1/AL/images";
//string trainConfFile = "/home/npillai1/AL/ConfFile/trainDataSet.conf";
//string trainConfFile = "/home/npillai1/AL/ConfFile/2kfiltered.conf";
//string trainConfFile = "/home/npillai1/AL/ConfFile/3kfiltered.conf";
//string trainConfFile = "/home/npillai1/AL/ConfFile/samplefiltered.conf";
//string trainConfFile = "/home/npillai1/AL/ConfFile/3k_tfidf_N10_filtered_fulldataset.conf";
//string trainConfFile = "/home/npillai1/AL/ConfFile/3k_Thresh_Lemmatized_fulldataset.conf";
//string trainConfFile = "/home/npillai1/AL/ConfFile/3k_tfidf_Thresh10.0_Lemmatized_fulldataset.conf";
//string trainConfFile = "/home/npillai1/AL/ConfFile/3k_filtered_fulldataset.conf";
string trainConfFile = "/home/npillai1/AL/ConfFile/3k_Thresh_fulldataset.conf";
//string trainConfFile = "/home/npillai1/AL/ConfFile/3k_tfidf_Thresh_Lemmatized_fulldataset.conf";
string trainLoc = imageLoc + "/trainImages";
string trainLocSyn = imageLoc + "/trainImagessyn";
string trainLocSynColor = imageLoc + "/trainImagesColor";
string trainLocSynShape= imageLoc + "/trainImagesShape";
string trainLocSynObj = imageLoc + "/trainImagesObject";
string testConfFile = "/home/npillai1/AL/ConfFile/testDataSet.conf";
string testLoc = imageLoc + "/testImages";
//string dsLoc = "/home/npillai1/AL/images/DS1";
string dsLoc = "/home/npillai1/AL/images/NmPx";
//string dsLoc = "/home/nish/DPD/thirdparty/kdes_2.0/bck-images/MTurkSet/rgbdcollection";
string libLinearLoc = "/home/npillai1/AL/kdes_2.0/liblinear-1.5-dense/";

//string testLocation = imageLoc + "/trainImages1kIII";
//string testLocation = imageLoc + "/UserStudy";
string testLocation = imageLoc + "/RSS-Batchmode-Traditional-I";
string trainLocation = testLocation;
// Function to delete prevous trained models and data


void deleteTrain() {
   string cmd = "rm -rf ";
   if(syn) {
      cmd += trainLocation ;
   } else {
      cmd += trainLocation ;
   }
   string log = "Clearing Existing trained data ...";
   voiceOut(log,false);
   int ret = system(cmd.c_str());
   
}

void trainFromFile() {
    if(syn) {
     parseFileAndTrain(dsLoc,trainConfFile,trainLocation,syn);  
    } else {
     parseFileAndTrain(dsLoc,trainConfFile,trainLocation,syn);
    }
}

void learnModel() {
    if(syn) {
       learn(trainLocation,syn);
    } else {
       learn(trainLocation,syn);
    }
}

void execute() {
 executeModel(trainLocation,libLinearLoc);

}

void alExecute() {
   alExecuteModel(dsLoc,trainLocation,trainConfFile,testConfFile,libLinearLoc,syn);
}

void alEntExecute() {
      alEntropyExecuteModel(dsLoc,trainLocation,trainConfFile,testConfFile,libLinearLoc,syn);
}
void mlBatchExecute() {
//	 mlBatchModel(dsLoc,trainLocation,trainConfFile,testConfFile,libLinearLoc,syn);
alEntropyExecuteModel(dsLoc,trainLocation,trainConfFile,testConfFile,libLinearLoc,syn);
}
void displayTestOption() {

} 

void testFromFile() {
   if(syn) {
       parseFileAndTest(dsLoc,testConfFile,trainLocation,syn);
   } else {
//    parseFileAndTest(dsLoc,testConfFile,testLoc);
       parseFileAndTest(dsLoc,testConfFile,trainLocation,syn);
   }
}

void saveDB() {

}
void createTrainLoc() {
  string cmd = "mkdir -p " + trainLocation;
  int ret = system(cmd.c_str());

}

void interactive()
{
  for (;;) {
      cout << ">> ";
      string input;
      getline(cin,input);
      if(input == "exit") {
        saveDB();
        exit(0);
      } else if (input == "live") {
         liveTraining = true;
      } else if (input == "file") {
        liveTraining = false;
      } else if(input == "clear") {
        deleteTrain();  
      } else if(input == "train") {
        if(liveTraining) {
        } else {
           trainFromFile();
           learnModel();
           testFromFile();
        }
      } else if(input == "test") {
         if(liveTraining) {
            displayTestOption();
         } else {
           testFromFile();
         }
      } else if (input == "learn") {
         learnModel();
      } else if(input == "execute") {
        cout << "exec" << endl;
        execute();
      } else if(input == "alex") {
        createTrainLoc();
        alExecute();
      } else if (input == "alent") {
        createTrainLoc();
        alEntExecute();
      }
  }    

}

int main ( int argc, char *argv[] )
{
   time_t now = time(0);
   char* dt = ctime(&now);
   cout << "The local date and time is: " << dt << endl;
   if ( argc == 1) {
      interactive();
   } else {
     string arg = string(argv[1]);
     if (arg == "clear") {
      deleteTrain();
     } else if (arg == "alent") {
      deleteTrain();
      createTrainLoc();
      alEntExecute();
      cout << "execution over" << endl;
     } else if (arg == "mlbatch") {
      createTrainLoc();
      mlBatchExecute();
      cout << "execution over" << endl;
     } else if(arg == "ch") {
	//justCheck();
     }
   }
   time_t now1 = time(0);
   char* dt1 = ctime(&now1);
   cout << "The local date and time is: " << dt1 << endl;
}
