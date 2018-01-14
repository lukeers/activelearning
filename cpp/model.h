#ifndef MODEL_H
#define MODEL_H

#include <string>
#include <vector>
using namespace std;
void justCheck();
void voiceOut(string cmd, bool speaker);
void parseFileAndTrain(string dsLoc,string trainConfFile,string trainLoc,bool syn);
void learn(string trainLoc,bool syn);
void parseFileAndTest(string dsLoc,string testConfFile,string testLoc,bool syn);
string executeModel(string trainLoc,string libLinearLoc);
void alExecuteModel(string dsLoc,string trainLocation,string trainConfFile,string testConfFile,string libLinearLoc,bool syn);
void onlineManualtrain(string croppedFiles,string speechLog);
vector<int> onlineAltrain(string objImgLoc,string trainLocation,string libLinearLoc);
void getLanguage(string spLoc,string spAudioLoc);
void alEntropyExecuteModel(string dsLoc,string trainLocation,string trainConfFile,string testConfFile,string libLinearLoc,bool syn);

void mlBatchModel(string dsLoc,string trainLocation,string trainConfFile,string testConfFile,string libLinearLoc,bool syn);

#endif
