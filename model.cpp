
#include "model.h"
#include "lang.h"
//#include "logit.h"
#include "logisticReg.h"
#include "logisticRegression.h"
#include <string>
#include <iostream>
#include <vector>
#include <set>
#include <fstream>
#include <map>
#include <algorithm>
#include <boost/filesystem.hpp>
#include <regex>
#include <math.h>
#include <queue>
#include <ctime>
#include <cstdlib>

using namespace boost::filesystem;
using namespace std;

int minInstances = 5;
string ckLoc = "/home/npillai1/AL/";
string testAnnot = ckLoc + "/ConfFile/testAnnotation.conf";
string fullAnnot = ckLoc + "/ConfFile/fullAnnotation.conf";
string colorAnnot = ckLoc + "/ConfFile/fullColorAnnotation.conf";
string shapeAnnot = ckLoc + "/ConfFile/fullShapeAnnotation.conf";
string objectAnnot = ckLoc + "/ConfFile/fullObjectAnnotation.conf";

string negPosInstances = "/home/npillai1/AL/negPosInstances.log";
map<string,vector<string> > labelIndMat;
map<string,vector<string> > posLabelIndMat;
string imageLoc1 = "/home/npillai1/AL/images";
string testFileLoc = imageLoc1 + "/testImages";

map<string,string> testFullAnnotation;
map<string,string> colorAnnotation;
map<string,string> shapeAnnotation;
map<string,string> objectAnnotation;
string wekaLoc = "/home/npillai1/AL/Weka/weka.jar";
string classifier = "weka.classifiers.functions.Logistic";

string kernelDir = "/home/npillai1/AL/kdes_2.0/demo_rgbd1/";
map<string,string> testAnnotation;

map<string,vector<string>> additionalNegInstances;
vector<string> unctFMeasurePoints;
void generateRgb(string rgb_file,string colorFile) {

	string command2 = "python /home/npillai1/AL/actl/src/findColorKMeans1.py --image \'" + colorFile + "\'  --clusters 3 > " + rgb_file;
	int ret3 = system(command2.c_str());

}

vector<string> getSpecFolders(string trFiles) {
	vector<string> fNames;
	path p(trFiles);
	for (auto i = directory_iterator(p); i != directory_iterator(); i++)
	{
		if (! is_directory(i->path())) //we eliminate directories
			//cout << i->path().filename().string() << endl;
			fNames.push_back(i->path().filename());
	}


	return fNames;
}

vector<string> getFolders(string trFiles) {
	vector<string> fNames;
	path p(trFiles);
	for (auto i = directory_iterator(p); i != directory_iterator(); i++)
	{
		if (is_directory(i->path())) //we eliminate directories
			//cout << i->path().filename().string() << endl;
			fNames.push_back(i->path().filename());
	}


	return fNames;
}

vector<string> split_string(const string& str,
		const string& delimiter)
{
	vector<string> strings;

	string::size_type pos = 0;
	string::size_type prev = 0;
	while ((pos = str.find(delimiter, prev)) != string::npos)
	{
		strings.push_back(str.substr(prev, pos - prev));
		prev = pos + 1;
	}

	// To get the last substring (or only, if delimiter is not found)
	strings.push_back(str.substr(prev));

	return strings;
}

void voiceOut(string cmd, bool speaker) {
	cout << cmd << endl;
	if(speaker) {
		string str = "echo \""+ cmd +"\"  |festival --tts";
		int ret = system(str.c_str());
	}
}

void posLabelMapUpdate(string lbl,string objImgLoc) {
	if(posLabelIndMat.find(lbl) == posLabelIndMat.end()) {
		posLabelIndMat[lbl];
	}
	posLabelIndMat[lbl].push_back(objImgLoc);
}

void negLabelMapUpdate(vector<string> antn1,string lbl) {
	if(labelIndMat.find(lbl) == labelIndMat.end()) {
		labelIndMat[lbl];
		for(map<string,vector<string>>::iterator itit=labelIndMat.begin();itit != labelIndMat.end();++itit) {
			ostringstream dat;
			dat << itit->first;
			string srt = dat.str();
			bool check = false;
			for(vector<string>::iterator itn = antn1.begin(); itn != antn1.end(); ++itn) {
				string lbla;
				lbla = *itn;
				if(lbla == srt) {
					check = true;
				}
			}

			if(check == false) {
				labelIndMat[lbl].push_back(srt);
				labelIndMat[srt].push_back(lbl);
			}
		}
	} else {
		vector<string> lbls = labelIndMat[lbl];
		vector<string> new_lbl;
		for(vector<string>::iterator itn = lbls.begin(); itn != lbls.end(); ++itn) {
			string lbla;
			lbla = *itn;
			bool check = false;
			for(vector<string>::iterator itn1 = antn1.begin(); itn1 != antn1.end(); ++itn1) {
				string lblb;
				lblb = *itn1;
				if(lblb == lbla) {
					check = true;
					vector<string>::iterator itA;
					itA = find(labelIndMat[lblb].begin(),labelIndMat[lblb].end(),lbl);
					labelIndMat[lblb].erase(itA);
				}
			}

			if(check == false) {
				new_lbl.push_back(lbla);
			}
		}
		labelIndMat.erase(lbl);
		labelIndMat[lbl] = new_lbl;
	}
}

void changeAndReplaceTName(string srcFldr,string itN,string destFldr,string dte) {
	string nn = itN + "_object.log";
	string newN = dte + "_object.log";
	string cmd = "cp -r " + srcFldr + "/" + nn + " " + destFldr + "/" + newN;
	int ret = system(cmd.c_str());
	nn = itN + "_shape.log";
	newN = dte + "_shape.log";
	cmd = "cp -r " + srcFldr + "/" + nn + " " + destFldr + "/" + newN;
	ret = system(cmd.c_str());
	//rename(srcFldr + "/" + nn, destFldr + "/" + newN);
	nn = itN + "_rgb.log";
	newN = dte + "_rgb.log";
	cmd = "cp -r " + srcFldr + "/" + nn + " " + destFldr + "/" + newN;
	ret = system(cmd.c_str());
	//rename(srcFldr + "/" + nn, destFldr + "/" + newN);
}
//          
void changeAndReplaceFName(string srcFldr,string itN,string destFldr,string dte) {
	string nn = itN + "_crop.png";
	string newN = dte + "_crop.png";
	rename(srcFldr + "/" + nn, destFldr + "/" + newN);
	nn = itN + "_depthcrop.png";
	newN = dte + "_depthcrop.png";
	rename(srcFldr + "/" + nn, destFldr + "/" + newN);
	nn = itN + "_maskcrop.png";
	newN = dte + "_maskcrop.png";
	rename(srcFldr + "/" + nn, destFldr + "/" + newN);
	nn = itN + "_loc.txt";
	newN = dte + "_loc.txt";
	rename(srcFldr + "/" + nn, destFldr + "/" + newN);
	nn = itN + "_rgb.log";
	newN = dte + "_rgb.log";
	//          rename(srcFldr + "/" + nn, destFldr + "/" + newN);
}
//
int copyTestFiles1(string objImgLoc,string newLabelLoc,int fldNo) {
	vector<string> files = getSpecFolders(objImgLoc);
	for(vector<string>::iterator itan = files.begin(); itan != files.end(); ++itan) {
		try {
			regex re("(.*)_(.*)_1_(.*)_crop.png");
			smatch match;
			if(regex_search(*itan,match,re) && match.size() > 1) {
				int newNo = 1;
				//    cout << match.str(1) << "--" << match.str(2) << "---" << match.str(3) << endl;
				ostringstream dte;
				//                       dte << newLabelLoc << "/test_1_1_" << newNo;
				dte << newLabelLoc << "/test_" << fldNo ;
				string cmd = "mkdir -p " + dte.str();
				int ret = system(cmd.c_str());
				dte << "/test_" << fldNo << "_1_" << newNo;
				string command = "cp " + objImgLoc  + "/" + match.str(1) + "_" + match.str(2) + "_1_" + match.str(3);
				string command2 = command + "_crop.png " + dte.str() + "_crop.png";
				int ret3 = system(command2.c_str());
				command2 = command + "_depthcrop.png " + dte.str() + "_depthcrop.png";
				ret3 = system(command2.c_str());
				command2 = command + "_maskcrop.png " + dte.str() + "_maskcrop.png";
				ret3 = system(command2.c_str());
				command2 = command + "_loc.txt " + dte.str() + "_loc.txt";
				ret3 = system(command2.c_str());
				string colorFile = dte.str() + "_crop.png";
				string rgb_file =  dte.str() + "_rgb.log";
				generateRgb(rgb_file,colorFile);
				//   cout << command2 << endl;
				newNo++;
				fldNo++;
			}
		} catch (regex_error& e) {
		}
	}
	return fldNo;
}

int copyTestFiles(string objImgLoc,string newLabelLoc,int fldNo) {
	vector<string> files = getSpecFolders(objImgLoc);
	for(vector<string>::iterator itan = files.begin(); itan != files.end(); ++itan) {
		string itName  = *itan;
		string chkName = "_crop.png";
		if (itName.find(chkName) != string::npos) {
			string itN = itName;
			ostringstream dte;
			dte << newLabelLoc << "/test_" << fldNo ;
			string cmd = "mkdir -p " + dte.str();
			int ret = system(cmd.c_str());
			ostringstream dte1;
			dte1 << "test_" << fldNo << "_1_1";
			int tSize = chkName.size();
			itN.replace(itN.end() - tSize,itN.end(),"");
			changeAndReplaceFName(objImgLoc,itN,dte.str(),dte1.str());
			fldNo++;      
		}
	}
	return fldNo;
}

int copyTestTFiles(string objImgLoc,string newLabelLoc,int fldNo) {
	vector<string> files = getSpecFolders(objImgLoc);
	for(vector<string>::iterator itan = files.begin(); itan != files.end(); ++itan) {
		string itName  = *itan;
		string chkName = "_rgb.log";
		if (itName.find(chkName) != string::npos) {
			string itN = itName;
			ostringstream dte;
			dte << newLabelLoc << "/test_" << fldNo ;
			string cmd = "mkdir -p " + dte.str();
			int ret = system(cmd.c_str());
			ostringstream dte1;
			dte1 << "test_" << fldNo;
			int tSize = chkName.size();
			itN.replace(itN.end() - tSize,itN.end(),"");
			changeAndReplaceTName(objImgLoc,itN,dte.str(),dte1.str());
			fldNo++;
		}
	}
	return fldNo;
}

void changeAndReplaceFName1(string srcFldr,string itN,string destFldr, string dte) {
	string nn = itN + "_crop.png";
	string newN = dte + "_crop.png";
	rename(srcFldr + "/" + nn, destFldr + "/" + newN);
	nn = itN + "_depthcrop.png";
	newN = dte + "_depthcrop.png";
	rename(srcFldr + "/" + nn, destFldr + "/" + newN);
	nn = itN + "_maskcrop.png";
	newN = dte + "_maskcrop.png";
	rename(srcFldr + "/" + nn, destFldr + "/" + newN);
	nn = itN + "_loc.txt";
	newN = dte + "_loc.txt";
	rename(srcFldr + "/" + nn, destFldr + "/" + newN);
	nn = itN + "_rgb.log";
	newN = dte + "_rgb.log";
	rename(srcFldr + "/" + nn, destFldr + "/" + newN);
}

void renameFilesAndCopy(string srcFldr,string destFldr ) {
	vector<string> names = split_string(srcFldr,"/");
	vector<string> newnames = split_string(destFldr,"/");
	string objN = names.at(names.size() - 1);
	string testN = newnames.at(newnames.size() - 1);
	int ln = objN.size();
	int fCount = 1;
	string cmd = "cp -r " + srcFldr + " " + destFldr;
	int ret = system(cmd.c_str());
	vector<string> files1 = getSpecFolders(destFldr);
	for(vector<string>::iterator itan = files1.begin(); itan != files1.end(); ++itan) {
		string itName  = *itan;
		string chkName = "_crop.png";
		if (itName.find(chkName) != string::npos) {
			string itN = itName;
			int tSize = chkName.size();
			ostringstream dte;
			dte <<  testN << "_1_" << fCount ;
			itN.replace(itN.end() - tSize,itN.end(),"");
			changeAndReplaceFName(destFldr,itN,destFldr,dte.str());
			fCount += 1;
		}
	}
	cout << destFldr << endl;
}

void renameTFilesAndCopy(string srcFldr,string destFldr ) {
	vector<string> names = split_string(srcFldr,"/");
	vector<string> newnames = split_string(destFldr,"/");
	string objN = names.at(names.size() - 1);
	string testN = newnames.at(newnames.size() - 1);
	int ln = objN.size();
	int fCount = 1;
	string cmd = "mkdir -p " + destFldr;
	int ret = system(cmd.c_str());
	vector<string> files1 = getSpecFolders(srcFldr);
	for(vector<string>::iterator itan = files1.begin(); itan != files1.end(); ++itan) {
		string itName  = *itan;
		string chkName = "_rgb.log";
		if (itName.find(chkName) != string::npos) {
			string itN = itName;
			int tSize = chkName.size();
			ostringstream dte;
			dte <<  testN ;
			itN.replace(itN.end() - tSize,itN.end(),"");
			changeAndReplaceTName(srcFldr,itN,destFldr,dte.str());
			fCount += 1;
		}
	}
}


void copyTrainFiles(string objImgLoc, vector<string> files,string fld,string tName) {
	string cmd = "rm -rf " + fld ;
	int ret = system(cmd.c_str());
	renameTFilesAndCopy(objImgLoc,fld);
}

void copyTrainFiles1(string objImgLoc, vector<string> files,string fld,string tName) {
	string cmd = "rm -rf " + fld + ";mkdir -p " + fld;
	int ret = system(cmd.c_str());
	cout << cmd << endl;
	int newNo = 1;
	//   cout << files.size() << endl;
	//   cout << objImgLoc << endl;
	vector<string> names = split_string(objImgLoc,"/");
	for (int i = 1;i < names.size() ; i++) {
		//   cout << names.at(i) << endl;
	}
	cout << tName << endl;
	cout << names.at(names.size()  - 1) << endl;
	for(vector<string>::iterator itan = files.begin(); itan != files.end(); ++itan) {
		try {
			regex re("(.*)_(.*)_1_(.*)_crop.png");
			smatch match;
			cout << "here " << *itan << endl;
			if(regex_search(*itan,match,re) && match.size() > 1) {
				cout << match.str(1) << "--" << match.str(2) << "---" << match.str(3) << endl;
				ostringstream dte;
				dte << fld << "/" << tName  << newNo;
				string command = "cp " + objImgLoc  + "/" + match.str(1) + "_" + match.str(2) + "_1_" + match.str(3);
				string command2 = command + "_crop.png " + dte.str() + "_crop.png";
				cout << command2 << endl;
				int ret3 = system(command2.c_str());
				command2 = command + "_depthcrop.png " + dte.str() + "_depthcrop.png";
				ret3 = system(command2.c_str());
				command2 = command + "_maskcrop.png " + dte.str() + "_maskcrop.png";
				ret3 = system(command2.c_str()); 
				command2 = command + "_loc.txt " + dte.str() + "_loc.txt";
				ret3 = system(command2.c_str());
				string colorFile = dte.str() + "_crop.png";
				string rgb_file =  dte.str() + "_rgb.log";
				generateRgb(rgb_file,colorFile);
				//   cout << command2 << endl;
				newNo++;
			}
		} catch (regex_error& e) {
		}
	}
}


vector<int> prepareTestFolders(string dsLoc,vector<string> cls,string tempTestLoc) {
	vector<int> fileIndexes;
	int fldNo = 1;
	int count = 1;
	fileIndexes.clear();
	string cmd = "rm -rf " + tempTestLoc + ";";
	cmd += "mkdir -p " + tempTestLoc;
	int ret = system(cmd.c_str());
	for (vector<string>::iterator itb = cls.begin() ; itb != cls.end(); ++itb) {
		string temp;
		temp = *itb;
		if(temp != "" ) {
			string objImgLoc = dsLoc + "/" + temp;
			int oldNo = fldNo;
			fldNo = copyTestTFiles(objImgLoc,tempTestLoc,fldNo);
			int newNo = fldNo;
			for(int k = oldNo; k < newNo; k++) {
				fileIndexes.push_back(count);
			}
		}
		count = count + 1;
	}
	return fileIndexes;
}


void parseAndCopySingleFile(string dsLoc, string trainConfFile,string trainLocation) {
	fstream fs(trainConfFile,fstream::in);
	string line;
	getline(fs,line,'\0');
	vector <string> cls = split_string(line,"\n");
	vector<string> cls1;
	for(unsigned i = 0; i <cls.size(); i++) {
		if(cls.at(i) != "") {
			vector<string> n = split_string(cls.at(i),",");
			cls1.push_back(n.front());
		}
	}
	string trLoc = trainLocation;
	vector<string> folders = getFolders(trainLocation);
	cout << "Adding Test Samples " << endl;
	string tempTestLoc = dsLoc + "/testtest";
	vector<int> fileIndexes = prepareTestFolders(dsLoc,cls1,tempTestLoc);
	for(vector<string>::iterator itan = folders.begin(); itan != folders.end(); ++itan) {
		string lbl ;
		lbl = *itan;
		string newLabelLoc = trainLocation + "/" + lbl + "/test";
		string cmd = "rm -rf " + newLabelLoc + ";";
		cmd += "cp -r " + tempTestLoc + " " +newLabelLoc;
		int ret = system(cmd.c_str());
	}

	cout << "Done :: Test samples added " << endl;
}



void parseAndCopySingleFile22(string dsLoc, string trainConfFile,string trainLoc) {
	fstream fs(trainConfFile,fstream::in);
	string line;
	getline(fs,line,'\0');
	vector <string> cls = split_string(line,"\n");
	string trLoc = trainLoc;
	vector<string> folders = getFolders(trainLoc);
	for(vector<string>::iterator itan = folders.begin(); itan != folders.end(); ++itan) {
		string lbl ;
		lbl = *itan;
		string newLabelLoc = trainLoc + "/" + lbl + "/"+lbl+"Pos/"+lbl+"Pos_1/" + lbl + "Pos_2.*";
		cout << newLabelLoc ;
		newLabelLoc = trainLoc + "/" + lbl + "/"+lbl+"Pos/"+lbl+"Pos_1";
		int newNo = 1;
		for (vector<string>::iterator itb = cls.begin() ; itb != cls.end(); ++itb) {
			string temp;
			temp = *itb;
			if(temp != "" ) {
				vector <string> feature = split_string(temp,",");
				string objImgLoc = dsLoc + "/" + feature.front();
				vector<string> files = getSpecFolders(objImgLoc);
				string annotation = feature.back();
				cout << objImgLoc <<" " << annotation << endl;

				for(vector<string>::iterator itan = files.begin(); itan != files.end(); ++itan) {
					try {
						regex re("(.*)_(.*)_1_(.*)_crop.png");
						smatch match;
						if(regex_search(*itan,match,re) && match.size() > 1) {
							//    cout << match.str(1) << "--" << match.str(2) << "---" << match.str(3) << endl;
							ostringstream dte;
							//                       dte << newLabelLoc << "/test_1_1_" << newNo;
							dte << newLabelLoc << "/" << lbl << "Pos_1_2_" << newNo;
							string command = "cp " + objImgLoc  + "/" + match.str(1) + "_" + match.str(2) + "_1_" + match.str(3);
							string command2 = command + "_crop.png " + dte.str() + "_crop.png";
							int ret3 = system(command2.c_str());
							command2 = command + "_depthcrop.png " + dte.str() + "_depthcrop.png";
							ret3 = system(command2.c_str());
							command2 = command + "_maskcrop.png " + dte.str() + "_maskcrop.png";
							ret3 = system(command2.c_str());
							command2 = command + "_loc.txt " + dte.str() + "_loc.txt";
							ret3 = system(command2.c_str());
							//   cout << command2 << endl;
							newNo++;
						}
					} catch (regex_error& e) {
					}
				}

			}

		}
	}
}

void parseAndCopySingleFile1(string dsLoc, string trainConfFile,string trainLoc) {
	//    int reqd = 1;
	fstream fs(trainConfFile,fstream::in);
	string line;
	getline(fs,line,'\0');
	vector <string> cls = split_string(line,"\n");
	string trLoc = trainLoc;
	//  int lineCount = 1;
	int newFNo = 1;
	vector<string> folders = getFolders(trainLoc);   
	for (vector<string>::iterator itb = cls.begin() ; itb != cls.end(); ++itb) {
		string temp;
		temp = *itb;
		if(temp != "" ) {
			//        if(lineCount == reqd) {
			vector <string> feature = split_string(temp,",");
			string objImgLoc = dsLoc + "/" + feature.front();
			vector<string> files = getSpecFolders(objImgLoc);
			string annotation = feature.back();
			cout << objImgLoc <<" " << annotation << endl;
			ostringstream dte1;
			dte1 <<  trLoc << newFNo;
			trainLoc = dte1.str();
			newFNo = newFNo + 1;
			string cm = "cp -r " + trLoc + " " + trainLoc;
			int ret1 = system(cm.c_str());


			for(vector<string>::iterator itan = folders.begin(); itan != folders.end(); ++itan) {
				string lbl ;
				lbl = *itan;
				string newLabelLoc = trainLoc + "/" + lbl + "/"+lbl+"Pos/"+lbl+"Pos_1/" + lbl + "Pos_2.*";
				cout << newLabelLoc ;
				//		string cmd = "rm -rf " + newLabelLoc;
				//                cout << cmd;
				//                int ret = system(cmd.c_str());
				newLabelLoc = trainLoc + "/" + lbl + "/"+lbl+"Pos/"+lbl+"Pos_1";
				//               newLabelLoc = trainLoc + "/" + lbl + "/test/test_1" ;  
				//                cout << "-----> " << lbl << "=====" << newLabelLoc << endl;
				string cmd = "mkdir -p " + newLabelLoc;
				int ret = system(cmd.c_str());
				vector<string> files2 = getSpecFolders(newLabelLoc);
				//                cout << files2.size() << endl;
				int newNo = files2.size() / 4;
				newNo = newNo + 1;
				cout << newNo << endl;
				for(vector<string>::iterator itan = files.begin(); itan != files.end(); ++itan) {
					try {
						regex re("(.*)_(.*)_1_(.*)_crop.png");
						smatch match;
						if(regex_search(*itan,match,re) && match.size() > 1) {
							//    cout << match.str(1) << "--" << match.str(2) << "---" << match.str(3) << endl;
							ostringstream dte;
							//                       dte << newLabelLoc << "/test_1_1_" << newNo;
							dte << newLabelLoc << "/" << lbl << "Pos_1_2_" << newNo; 
							string command = "cp " + objImgLoc  + "/" + match.str(1) + "_" + match.str(2) + "_1_" + match.str(3);
							string command2 = command + "_crop.png " + dte.str() + "_crop.png";
							int ret3 = system(command2.c_str());
							command2 = command + "_depthcrop.png " + dte.str() + "_depthcrop.png";
							ret3 = system(command2.c_str());
							command2 = command + "_maskcrop.png " + dte.str() + "_maskcrop.png";
							ret3 = system(command2.c_str());
							command2 = command + "_loc.txt " + dte.str() + "_loc.txt";
							ret3 = system(command2.c_str());
							//   cout << command2 << endl;
							newNo++;
						}
					} catch (regex_error& e) {
					}
				}

			}

			//	      }
			//       lineCount ++;
		}
	}

}

void fileExactCopy(string dest,string wrd, string objImgLoc) {
	vector<string> files2 = getFolders(dest);
	int nNo = files2.size();
	int fldrNo = nNo + 1;
	int newNo = 1;
	ostringstream dte; 
	dte << dest << "/" << wrd << "Pos_" << fldrNo;
	renameFilesAndCopy(objImgLoc,dte.str());
	/*
	   string cmd = "cp -r " + objImgLoc + " " + dte.str();
	   int ret = system(cmd.c_str());
	   ostringstream dt;
	   dt << wrd << "Pos_" << fldrNo << "_1_";
	   string s = "[a-z]+_[\\d]+_1_";
	   vector<string> files1 = getSpecFolders(dte.str());
	   for(vector<string>::iterator itan1 = files1.begin(); itan1 != files1.end(); ++itan1) {
	   string itName  = *itan1;
	   int ln = s.size();
	   itName.replace(itName.begin(),itName.begin() + ln,dt.str());
	   rename(dte.str() + "/" + *itan1, dte.str() + "/" + itName);
	   }
	   cout << dte.str() << endl;
	//	cmd = "cd " + dte.str() + ";rename 's/" + s + "/" + dt.str() + "/i' *";
	//	ret = system(cmd.c_str());
	*/
}


void filesCopy(string dest,string wrd, string objImgLoc) {
	vector<string> fSrc = getFolders(objImgLoc);
	string objIm = objImgLoc;
	for(vector<string>::iterator itn = fSrc.begin(); itn != fSrc.end(); ++itn) {
		ostringstream dte1;
		dte1 << objIm << "/" << *itn;
		objImgLoc = dte1.str();
		fileExactCopy(dest,wrd, objImgLoc);
	}
}

void filesCopy1(string dest,string wrd, string objImgLoc) {
	vector<string> files2 = getSpecFolders(dest);
	//                cout << files2.size() << endl;
	int newNo = files2.size() / 4;
	newNo = newNo + 1;
	vector<string> files = getSpecFolders(objImgLoc);
	for(vector<string>::iterator itan = files.begin(); itan != files.end(); ++itan) {
		try {
			regex re("(.*)_(.*)_1_(.*)_crop.png");
			smatch match;
			if(regex_search(*itan,match,re) && match.size() > 1) {
				//    cout << match.str(1) << "--" << match.str(2) << "---" << match.str(3) << endl;
				ostringstream dte;
				dte << dest << "/" << wrd << "Pos_1_1_" << newNo;
				string command = "cp " + objImgLoc  + "/" + match.str(1) + "_" + match.str(2) + "_1_" + match.str(3);
				string command2 = command + "_crop.png " + dte.str() + "_crop.png";
				int ret3 = system(command2.c_str());
				command2 = command + "_depthcrop.png " + dte.str() + "_depthcrop.png";
				ret3 = system(command2.c_str());
				command2 = command + "_maskcrop.png " + dte.str() + "_maskcrop.png";
				ret3 = system(command2.c_str());
				command2 = command + "_loc.txt " + dte.str() + "_loc.txt";
				ret3 = system(command2.c_str());
				//   cout << command2 << endl;
				newNo++;
			}
		} catch (regex_error& e) {
		}
	}
}

void addSynImage(string trainLoc,string lbl,string objImgLoc) {
	vector<string> files1 = getFolders(trainLoc);
	for(vector<string>::iterator itan = files1.begin(); itan != files1.end(); ++itan) {
		if ((*itan).find("syn") != string::npos) {
			if ((*itan).find(lbl) != string::npos) {
				string wrd = *itan;
				string newLabelLoc = trainLoc + "/" + wrd + "/"+wrd+"Pos/";
				fileExactCopy(newLabelLoc,wrd, objImgLoc);
			}
		}
		/*
		   try {
		   regex re("syn-(.*)" + lbl + "(.*)");
		   smatch match;
		   if(regex_search(*itan,match,re) && match.size() > 1) {
		   string wrd = *itan;
		   string newLabelLoc = trainLoc + "/" + wrd + "/"+wrd+"Pos/";
//          filesCopy(newLabelLoc,wrd,objImgLoc);
fileExactCopy(newLabelLoc,wrd, objImgLoc); 
}
} catch (regex_error& e) {
}
*/
}


}

void newSynCreate(string trainLoc,string lbl) {
	for(map<string,vector<string>>::iterator itit=labelIndMat.begin();itit != labelIndMat.end();++itit) {
		ostringstream dat;
		dat << itit->first;
		string srt = dat.str();
		if(srt != lbl) {
			string wrd = "syn-" + lbl + "-" + srt;
			string newLabelLoc = trainLoc + "/" + wrd + "/"+wrd+"Pos/";
			string cmd = "mkdir -p " + newLabelLoc;
			int ret = system(cmd.c_str());
			string objImgLoc = trainLoc + "/" + srt + "/"+srt+"Pos/";
			filesCopy(newLabelLoc,wrd,objImgLoc);
			objImgLoc = trainLoc + "/" + lbl + "/"+lbl+"Pos/";
			filesCopy(newLabelLoc,wrd,objImgLoc);
		}
	}

}
void parseFileAndCopyImages22(string dsLoc, string trainConfFile,string trainLoc,bool negLabelUpdate,bool syn) {
	//    cout << trainLoc << endl;
	fstream fs(trainConfFile,fstream::in);
	string line;
	getline(fs,line,'\0');
	vector <string> cls = split_string(line,"\n");
	for (vector<string>::iterator itb = cls.begin() ; itb != cls.end(); ++itb) {
		string temp;
		temp = *itb;
		if(temp != "") {
			vector <string> feature = split_string(temp,",");
			string objImgLoc = dsLoc + "/" + feature.front();
			string destImgLoc = trainLoc + "/" + feature.front();
			string cmd = "mkdir -p " + destImgLoc;
			int ret = system(cmd.c_str());  
			cmd = "cp -r " + objImgLoc + "/*.* " + destImgLoc + "/";
			ret = system(cmd.c_str());
		}
	}
	string fileN = "/home/nish/catkin_ws/ConfFile/testDataSet.conf";
	fstream fs1(fileN,fstream::in);
	getline(fs1,line,'\0');
	cls = split_string(line,"\n");
	int newNo = 1;
	for (vector<string>::iterator itb = cls.begin() ; itb != cls.end(); ++itb) {
		string temp;
		temp = *itb;
		if(temp != "") {
			vector <string> feature = split_string(temp,",");
			string objImgLoc = dsLoc + "/" + feature.front();
			vector<string> files = getSpecFolders(objImgLoc);
			string destImgLoc = trainLoc + "/test/arch/arch_1";
			string cmd = "mkdir -p " + destImgLoc;
			int ret = system(cmd.c_str());
			for(vector<string>::iterator itan = files.begin(); itan != files.end(); ++itan) {
				try {
					regex re("(.*)_(.*)_1_(.*)_crop.png");
					smatch match;
					if(regex_search(*itan,match,re) && match.size() > 1) {
						//    cout << match.str(1) << "--" << match.str(2) << "---" << match.str(3) << endl;
						ostringstream dte;
						dte << destImgLoc << "/arch_1_2_" << newNo;
						string command = "cp " + objImgLoc  + "/" + match.str(1) + "_" + match.str(2) + "_1_" + match.str(3);
						string command2 = command + "_crop.png " + dte.str() + "_crop.png";
						int ret3 = system(command2.c_str());
						command2 = command + "_depthcrop.png " + dte.str() + "_depthcrop.png";
						ret3 = system(command2.c_str());
						command2 = command + "_maskcrop.png " + dte.str() + "_maskcrop.png";
						ret3 = system(command2.c_str());
						command2 = command + "_loc.txt " + dte.str() + "_loc.txt";
						ret3 = system(command2.c_str());
						//   cout << command2 << endl;
						newNo++;
					}
				} catch (regex_error& e) {
				}
			}
		}
	}

}

void addtrainFolderUnderAnnotation(string objImgLoc,string trainLoc,vector<string> anntn,vector<string> files,bool negLabelUpdate,bool syn) {
	vector<string> antn1;
	antn1 = anntn; 
	string tName = "testtest_1";
	string fld = trainLoc + "/" + tName;
	tName += "_1_";
	copyTrainFiles(objImgLoc, files,fld,tName);

	for(vector<string>::iterator itan = anntn.begin(); itan != anntn.end(); ++itan) {
		string lbl ;
		lbl = *itan;
		cout << lbl << endl;
		string newLabelLoc = trainLoc + "/" + lbl + "/"+lbl+"Pos/";
		string cmd = "mkdir -p " + newLabelLoc;
		int ret = system(cmd.c_str());

		vector<string> files2 = getFolders(newLabelLoc);
		int newNo = files2.size();
		int newFldr = newNo + 1;

		ostringstream dte1;
		dte1 << newLabelLoc << lbl << "Pos_" << newFldr;
		newLabelLoc = dte1.str();
		renameFilesAndCopy(fld,newLabelLoc);
		/*
		//     cmd = "mkdir -p " + newLabelLoc;
		cmd = "cp -r " + fld + " " + newLabelLoc;
		ostringstream dt;
		dt << lbl << "Pos_" << newFldr << "_1_";
		//   cout << cmd << endl;
		ret = system(cmd.c_str());

		vector<string> files1 = getSpecFolders(newLabelLoc);
		for(vector<string>::iterator itan1 = files1.begin(); itan1 != files1.end(); ++itan1) {
		string itName  = *itan1;
		int ln = tName.size();
		itName.replace(itName.begin(),itName.begin() + ln,dt.str());
		rename(newLabelLoc + "/" + *itan1, newLabelLoc + "/" + itName);
		}
		cout << newLabelLoc << endl;
		//      cmd = "rename 's/" + tName + "/" + dt.str() + "/' " + newLabelLoc + "/*";   
		//      cout << cmd << endl;
		//      ret = system(cmd.c_str());
		*/
		if(newFldr == 1 && syn) {
			newSynCreate(trainLoc,lbl);
		} else if(syn) {
			addSynImage(trainLoc,lbl,objImgLoc);
		}

		if(negLabelUpdate) {
			negLabelMapUpdate(antn1,lbl);
		}

	}
	string cmd = "rm -rf " + fld;
	int ret = system(cmd.c_str());
}

void parseFileAndCopyImages(string dsLoc, string trainConfFile,string trainLoc,bool negLabelUpdate,bool syn) {
	//    cout << trainLoc << endl;
	fstream fs(trainConfFile,fstream::in);
	string line;
	getline(fs,line,'\0');
	vector <string> cls = split_string(line,"\n");
	for (vector<string>::iterator itb = cls.begin() ; itb != cls.end(); ++itb) {
		string temp;
		temp = *itb;
		if(temp != "") {
			vector <string> feature = split_string(temp,",");
			string objImgLoc = dsLoc + "/" + feature.front();
			vector<string> files = getSpecFolders(objImgLoc);
			string annotation = feature.back();
			cout << objImgLoc <<" " << annotation << endl;
			vector<string> anntn = processLanguage(annotation);
			addtrainFolderUnderAnnotation(objImgLoc,trainLoc,anntn,files,true,syn);
		}
	}
}


void parseFileAndCopyImages11(string dsLoc, string trainConfFile,string trainLoc,bool negLabelUpdate,bool syn) {
	//    cout << trainLoc << endl;
	fstream fs(trainConfFile,fstream::in);
	string line;
	getline(fs,line,'\0');
	vector <string> cls = split_string(line,"\n");
	for (vector<string>::iterator itb = cls.begin() ; itb != cls.end(); ++itb) {
		string temp;
		temp = *itb;
		if(temp != "") {
			vector <string> feature = split_string(temp,",");
			string objImgLoc = dsLoc + "/" + feature.front();
			vector<string> files = getSpecFolders(objImgLoc);
			//             cout << files.size() << endl;
			string annotation = feature.back();
			cout << objImgLoc <<" " << annotation << endl;
			vector<string> anntn = processLanguage(annotation);
			vector<string> antn1;
			antn1 = anntn;
			for(vector<string>::iterator itan = anntn.begin(); itan != anntn.end(); ++itan) {
				string lbl ;
				lbl = *itan;
				string newLabelLoc = trainLoc + "/" + lbl + "/"+lbl+"Pos/"+lbl+"Pos_1";

				//                cout << "-----> " << lbl << "=====" << newLabelLoc << endl;
				string cmd = "mkdir -p " + newLabelLoc;
				int ret = system(cmd.c_str());
				vector<string> files2 = getSpecFolders(newLabelLoc);
				//                cout << files2.size() << endl;
				int newNo = files2.size() / 4;
				int newLabel = newNo;
				newNo = newNo + 1;
				cout << newNo << endl;
				for(vector<string>::iterator itan = files.begin(); itan != files.end(); ++itan) {
					try {
						regex re("(.*)_(.*)_1_(.*)_crop.png");
						smatch match;
						if(regex_search(*itan,match,re) && match.size() > 1) {
							//    cout << match.str(1) << "--" << match.str(2) << "---" << match.str(3) << endl;
							ostringstream dte;
							dte << newLabelLoc << "/" << lbl << "Pos_1_1_" << newNo;
							string command = "cp " + objImgLoc  + "/" + match.str(1) + "_" + match.str(2) + "_1_" + match.str(3);
							string command2 = command + "_crop.png " + dte.str() + "_crop.png";
							int ret3 = system(command2.c_str());
							command2 = command + "_depthcrop.png " + dte.str() + "_depthcrop.png";
							ret3 = system(command2.c_str());
							command2 = command + "_maskcrop.png " + dte.str() + "_maskcrop.png";
							ret3 = system(command2.c_str());
							command2 = command + "_loc.txt " + dte.str() + "_loc.txt";
							ret3 = system(command2.c_str());
							//   cout << command2 << endl;
							newNo++;
						}
					} catch (regex_error& e) {
					}
				}
				if(newLabel == 0 && syn) {
					newSynCreate(trainLoc,lbl);
				} else if(syn) {
					addSynImage(trainLoc,lbl,objImgLoc);
				}

				if(negLabelUpdate) {
					negLabelMapUpdate(antn1,lbl);
				}
			}
		}
	}


}

void parseFileAndTrain(string dsLoc, string trainConfFile,string trainLoc,bool syn) {
	parseFileAndCopyImages(dsLoc,trainConfFile,trainLoc,true,syn);
}

void parseFileAndTest(string dsLoc,string testConfFile,string testLoc,bool syn) {
	//      parseFileAndCopyImages(dsLoc,testConfFile,testLoc,false);
	parseAndCopySingleFile(dsLoc,testConfFile,testLoc);
}

void addTNegTFIDF(string trainLoc, string srt,vector<string> labelInd) {
	string fLoc = trainLoc + "/" + srt + "/" + srt + "Pos/";
	if (! exists(fLoc))
		return;
	vector<string> fCount = getFolders(fLoc);
	int countNeg  = fCount.size();
	string newLabelLoc = trainLoc + "/" + srt + "/"+srt+"Neg/";
	string cmd = "rm -rf "+ newLabelLoc + ";mkdir -p " + newLabelLoc;
	int ret = system(cmd.c_str());
	int cNo = 1;
	int negLabelCount = 0;
	int roundC = 0;
	int emptyCount = 0;
	int negCountMore = labelInd.size() / 2;
	//   cout << "-----> " << srt << " ";
	for (int i = 0;i < labelInd.size(); i++) {
		//    cout << labelInd.at(i) << ", ";
	}
	//   cout << endl;
	ofstream myfile;
	myfile.open(negPosInstances,std::ios_base::app);
	myfile << endl << "Negative Instances :: ";
	//   while((cNo <= (countNeg + labelInd.size())) and (emptyCount < labelInd.size())) {

//	while((cNo <= countNeg) and (emptyCount < labelInd.size())) {
	for (int i = 0;i < labelInd.size(); i++) {
		roundC++;
		unsigned clock = roundC % labelInd.size();
//		string lbla = labelInd.at(clock);
		string lbla = labelInd.at(i);
		//         cout <<"neg up " <<  lbla << endl;
		if(exists(lbla)) {
			myfile  << lbla << ", ";
			emptyCount = 0;
			ostringstream dte;
			dte << newLabelLoc << "/" << srt << "Neg_" << cNo;
			renameTFilesAndCopy(lbla,dte.str());
			cNo++;
		} else {
			emptyCount++;
		}

	}
	myfile << endl; 
	myfile.close(); 
	cout << "Negative count " << labelInd.size() << endl;
}

void addTNeg(string trainLoc, string srt,vector<string> labelInd) {
	string fLoc = trainLoc + "/" + srt + "/" + srt + "Pos/";
	if (! exists(fLoc))
		return;
	vector<string> fCount = getFolders(fLoc);
	int countNeg  = fCount.size();
	string newLabelLoc = trainLoc + "/" + srt + "/"+srt+"Neg/";
	string cmd = "rm -rf "+ newLabelLoc + ";mkdir -p " + newLabelLoc;
	int ret = system(cmd.c_str());
	int cNo = 1;
	int negLabelCount = 0;
	int roundC = 0;
	int emptyCount = 0;
	int negCountMore = labelInd.size() / 2;
	while((cNo <= (countNeg + labelInd.size())) and (emptyCount < labelInd.size())) {
		//   while((cNo <= (countNeg + negCountMore)) and (emptyCount < labelInd.size())) {
		roundC++;
		unsigned clock = roundC % labelInd.size();
		string lbla = labelInd.at(clock);
		negLabelCount = (roundC / labelInd.size() ) - 1;
		ostringstream dtete;
		//   dtete << trainLoc << "/" << lbla << "/" << lbla << "Pos/" << lbla << "Pos_" << negLabelCount;
		vector<string> posNeg = posLabelIndMat[lbla];
		if (negLabelCount < posNeg.size()) {    
			//      string crpFile = dtete.str();
			//      if(exists(crpFile)) {
			string crpFile = posNeg.at(negLabelCount);
			emptyCount = 0;
			ostringstream dte;
			dte << newLabelLoc << "/" << srt << "Neg_" << cNo;
			renameTFilesAndCopy(crpFile,dte.str());
			cNo++;
		} else {
			emptyCount++;
		}
		}

	}

	void addNeg(string trainLoc, string srt,vector<string> labelInd) {
		cout << srt << endl;
		string fLoc = trainLoc + "/" + srt + "/" + srt + "Pos/";
		if (! exists(fLoc)) 
			return;
		vector<string> fCount = getFolders(fLoc);
		int countNeg  = fCount.size();
		string newLabelLoc = trainLoc + "/" + srt + "/"+srt+"Neg/";
		string cmd = "rm -rf "+ newLabelLoc + ";mkdir -p " + newLabelLoc;
		int ret = system(cmd.c_str());
		int cNo = 1;
		int negLabelCount = 0;
		int roundC = 0;
		int emptyCount = 0;
		while((cNo <= (countNeg + labelInd.size())) and (emptyCount < labelInd.size())) {
			//   while((cNo <= countNeg) and (emptyCount < labelInd.size())) {
			roundC++;
			unsigned clock = roundC % labelInd.size();
			string lbla = labelInd.at(clock);
			negLabelCount = roundC / labelInd.size();
			ostringstream dtete;
			dtete << trainLoc << "/" << lbla << "/" << lbla << "Pos/" << lbla << "Pos_" << negLabelCount;
			string crpFile = dtete.str();    
			if(exists(crpFile)) {
				emptyCount = 0;
				ostringstream dte;
				dte << newLabelLoc << "/" << srt << "Neg_" << cNo;
				renameFilesAndCopy(crpFile,dte.str());
				/*
				   cmd = "cp -r " + crpFile + " " + dte.str();
				   ret = system(cmd.c_str());
				   cout << cmd << endl;
				   ostringstream dtdt1;
				   dtdt1 <<  lbla << "Pos_" << negLabelCount << "_1_";
				   ostringstream dtdt2;
				   dtdt2 <<   srt << "Neg_" << cNo << "_1_";

				   cmd = "rename 's/" + dtdt1.str() + "/" + dtdt2.str() + "/' " + dte.str() + "/*";
				   cout << cmd << endl;
				   ret = system(cmd.c_str());

*/


				/*	 vector<string> files2 = getSpecFolders(crpFile);
					 int newNo = 1;
					 for(vector<string>::iterator itan = files2.begin(); itan != files2.end(); ++itan) {
					 try {
					 regex re("(.*)_(.*)_1_(.*)_crop.png");
					 smatch match;
					 if(regex_search(*itan,match,re) && match.size() > 1) {
					 ostringstream dte;
					 dte << newLabelLoc << "/" << srt << "Neg_" << cNo;
					 cmd = "mkdir -p " + dte.str();
					 ret = system(cmd.c_str());
					 dte <<  "/" << srt << "Neg_" << cNo << "_1_" << newNo;
					 string command = "cp " + crpFile  + "/" + match.str(1) + "_" + match.str(2) + "_1_" + match.str(3);
					 string command2 = command + "_crop.png " + dte.str() + "_crop.png";
					 int ret3 = system(command2.c_str());
					 command2 = command + "_depthcrop.png " + dte.str() + "_depthcrop.png";
					 ret3 = system(command2.c_str());
					 command2 = command + "_maskcrop.png " + dte.str() + "_maskcrop.png";
					 ret3 = system(command2.c_str());
					 command2 = command + "_loc.txt " + dte.str() + "_loc.txt";
					 ret3 = system(command2.c_str());
				//   cout << command2 << endl;
				string colorFile = dte.str() + "_crop.png";
				string rgb_file =  dte.str() + "_rgb.log";
				generateRgb(rgb_file,colorFile);
				newNo++;
				}
				} catch (regex_error& e) {
				}
				} */
				cNo++;
			} else {
				emptyCount++;
			}
		}

		}


		void addNeg22(string trainLoc, string srt,vector<string> labelInd) {

			string fLoc = trainLoc + "/" + srt + "/"+srt+"Pos/"+srt+"Pos_1";
			vector<string> fCount = getSpecFolders(fLoc);
			int countNeg  = fCount.size() / 4;

			string newLabelLoc = trainLoc + "/" + srt + "/"+srt+"Neg/"+srt+"Neg_1";
			cout << newLabelLoc << endl;
			string cmd = "mkdir -p " + newLabelLoc;
			int ret = system(cmd.c_str());
			int cNo = 1;
			int negLabelCount = 0;
			int roundC = 0;
			int emptyCount = 0;
			while((cNo < countNeg) and (emptyCount < labelInd.size())) {
				roundC++;
				unsigned clock = roundC % labelInd.size();
				string lbla = labelInd.at(clock);
				negLabelCount = roundC / labelInd.size();

				string negLoc = trainLoc + "/" + lbla + "/"+lbla+"Pos/"+lbla+"Pos_1/" + lbla + "Pos_1_1_";
				ostringstream dtet;
				dtet << negLoc << negLabelCount << "_";
				string crpFile = dtet.str() + "crop.png";
				cout << crpFile << endl;
				if(exists(crpFile)) {
					emptyCount = 0;
					ostringstream dte;
					newLabelLoc = trainLoc + "/" + srt + "/"+srt+"Neg/"+srt+"Neg_1";
					dte << newLabelLoc << "/" << srt << "Neg_1_1_" << cNo;
					string command2 = "cp " + crpFile + " " + dte.str() + "_crop.png";
					int ret3 = system(command2.c_str());
					crpFile = dtet.str() + "depthcrop.png ";
					command2 = "cp " + crpFile + dte.str() + "_depthcrop.png";
					ret3 = system(command2.c_str());
					crpFile = dtet.str() + "maskcrop.png ";
					command2 = "cp " + crpFile + dte.str() + "_maskcrop.png";
					ret3 = system(command2.c_str());
					crpFile = dtet.str() + "loc.txt ";
					command2 = "cp " + crpFile + dte.str() + "_loc.txt";
					ret3 = system(command2.c_str());
					cNo++;
				} else {
					emptyCount++;
				}
			}

		}

		void synNegAddFiles(string trainLoc) {
			int countNeg;
			vector<string> files1 = getFolders(trainLoc);
			for(vector<string>::iterator itan1 = files1.begin(); itan1 != files1.end(); ++itan1) {
				try {
					regex re("syn-(.*)-(.*)");
					smatch match;
					if(regex_search(*itan1,match,re) && match.size() > 1) {
						string srt = *itan1;

						string lbl = match.str(1);
						string lbl2 = match.str(2);
						if(lbl == "" || lbl2 == "") {
							continue;
						}
						vector<string> v1 = labelIndMat[lbl];
						vector<string> v2 = labelIndMat[lbl2];
						vector<string> v3;
						if(v1.size() == 0 || v2.size() == 0) {
							continue;
						}
						sort(v1.begin(), v1.end());
						sort(v2.begin(), v2.end());

						set_intersection(v1.begin(),v1.end(),v2.begin(),v2.end(),back_inserter(v3));
						vector<string> v;
						copy(v3.begin(),v3.end(),inserter(v,v.begin()));
						addNeg(trainLoc, srt,v);
					}
				} catch (regex_error& e) {
				}
			}
		}

		void learn(string trainLoc,bool syn) {
			cout << "Adding Negative Examples" << endl;
			for(map<string,vector<string>>::iterator itit=labelIndMat.begin();itit != labelIndMat.end();++itit) {
				ostringstream dat;
				dat << itit->first;
				string srt = dat.str();
				//    string srt = *itit;
				addNeg(trainLoc, srt,labelIndMat[srt]);
			}
			if(syn) {
				synNegAddFiles(trainLoc);
			}
			cout << "Done :: Negative Examples Added " << endl;
		}

		void executeRGBModel(string trainLoc,string libLinearLoc,string rgbCSVFile) {
			vector<string> folders = getFolders(trainLoc +"/");

			string tFile  = trainLoc + "/temp.csv";
			string cm = " >"+tFile;
			int ret1 = system(cm.c_str());
			string testInstances;
			string probabilities = "";
			for(vector<string>::iterator itan = folders.begin(); itan != folders.end(); ++itan) {
				string classfr = *itan;
				cout << "-----------------------" << classfr << "----------------------------------";
				string trainSet = trainLoc + "/" + *itan + "/trainSet.linear";
				string testSet = trainLoc + "/" + *itan + "/testSet.linear";
				string modelFile = trainLoc + "/" + *itan + "/" + *itan + "-classification.model";
				string outputFile = trainLoc + "/" + *itan + "/" + *itan + "-out.file";
				string str = "rm -rf "+ trainSet;
				int ret = system(str.c_str());
				str = "rm -rf "+ testSet;
				ret = system(str.c_str());

				vector<string> subFolders = getFolders(trainLoc + "/" + *itan + "/");       
				string outFile;
				string label;
				testInstances = "";
				for(vector<string>::iterator itsub = subFolders.begin(); itsub != subFolders.end(); ++itsub) {
					string clsName = *itsub;

					if(clsName == "test") {
						label = "1 ";
						outFile = testSet;
						testInstances = "";
					} else if(clsName == *itan+"Pos") {
						label = "1 ";
						outFile = trainSet;
					} else if(clsName == *itan+"Neg") {
						label = "-1 ";
						outFile = trainSet;
					}
					cout << clsName << endl;
					vector<string> clsFolders = getFolders(trainLoc + "/" + *itan + "/" + clsName + "/");

					for(vector<string>::iterator itClssub = clsFolders.begin(); itClssub != clsFolders.end(); ++itClssub) {
						string clsInstName = *itClssub;
						vector<string> imgFiles = getSpecFolders(trainLoc + "/" + *itan + "/"  + clsName + "/" + clsInstName + "/");

						for(vector<string>::iterator img = imgFiles.begin(); img != imgFiles.end(); ++img) {
							string imgName = *img;
							if (imgName.find("_rgb.log") != string::npos) {
								//                  try {
								//                     regex re("(.*)_rgb.log");
								//                     smatch match;
								//                     if(regex_search(imgName,match,re) && match.size() > 1) {
								if(clsName == "test") {
									testInstances += clsInstName + ", ";
								}

								string imNm = trainLoc + "/" + *itan + "/" + clsName + "/" + "/" + clsInstName + "/" + imgName;
								string command = "echo -n \'" + label + "\' >> "+ outFile +";cat "+imNm + " >> "+ outFile;
								ret = system(command.c_str());
							} 
							//
							//		  } catch (regex_error& e) {
							//		  }
							}
						}
					}
					cout << "1" << endl;
					cout << trainSet << " " << modelFile<< endl;
					string cmd1 = libLinearLoc + "/train -s 0 -c 10 " + trainSet + " " + modelFile;
					ret = system(cmd1.c_str());
					cout << "2" << endl;
					cmd1 = libLinearLoc + "/predict -b 1 " + testSet + " " + modelFile + " " + outputFile;
					ret = system(cmd1.c_str());
					cout << "3" << endl;
					cout << modelFile << " " << outputFile << endl;
					fstream fs(outputFile,fstream::in);
					string line;
					getline(fs,line,'\0');
					vector <string> predictions = split_string(line,"\n");
					unsigned posIndex = 1;
					probabilities = "";
					for (vector<string>::iterator pred = predictions.begin() ; pred != predictions.end(); ++pred) {
						string clsPred = *pred;
						cout << "clsPred------> " << endl;
						cout << clsPred << endl;
						try {
							regex re("labels (.*) (.*)");
							smatch match;
							if(regex_search(clsPred,match,re) && match.size() > 1) {
								if(match.str(2) == "1") {
									posIndex = 2;
								}
							} else {
								vector<string> probs = split_string(clsPred," ");
								if(probs.size() > 2) {
									probabilities += probs.at(posIndex) + ", ";
								}
							}
						} catch(regex_error& e) {
						}

					}
					cmd1 = "echo label," + testInstances + " >> " + tFile + ";echo " + classfr + "," + probabilities +">>" + tFile;      
					//        cmd1 = "echo " + classfr + "," + probabilities +">>" + tFile;
					ret = system(cmd1.c_str());
				}
				//   cout << testInstances << endl;
				string cmd = "echo label," + testInstances + " > " + rgbCSVFile + "; cat "+ tFile +" >> " + rgbCSVFile;
				int ret = system(cmd.c_str());
				cout << rgbCSVFile << endl;
			}

			string executeModel(string trainLoc,string libLinearLoc) {
				string rgbCSVFile = trainLoc + "/rgbConfusionMatrix.csv"; 
				string cmd = ">"+ rgbCSVFile;
				int ret = system(cmd.c_str());
				cout << "-------RGB CSV FILE--------------- " << rgbCSVFile; 
				executeRGBModel(trainLoc,libLinearLoc,rgbCSVFile);
				return rgbCSVFile;
			}

			void testWithTestFolder(string shapeCSVFile, string objectCSVFile,string trainLocation,string fName) {
				string arg = kernelDir + "argmnts.log";
				string cmd = ">"+ shapeCSVFile;
				int ret = system(cmd.c_str());
				cmd = ">"+ objectCSVFile;
				ret = system(cmd.c_str());
				//    string fName = "testing";
				string fN = trainLocation + "  " + shapeCSVFile + " " + objectCSVFile;
				cmd = "echo " + fN + " > " + arg;
				ret = system(cmd.c_str());

				cmd = "cd " + kernelDir + ";matlab -nosplash -nodesktop -r " + fName;
				cmd += ",quit force,quit,exit";
				cout << cmd << endl;
				ret = system(cmd.c_str());
				cmd = "cat " + shapeCSVFile ;
				//    ret = system(cmd.c_str());
				cmd = "cat " + objectCSVFile ;
				//   ret = system(cmd.c_str());


			}

			vector<double> calculateEntropy(vector<double> probSamples) {
				vector<double> entrps;
				double maxUt = 0.0;
				for (vector<double>::iterator entr = probSamples.begin() ; entr != probSamples.end(); ++entr) {
					double val = *entr;
					double utMesh = log2(val) * val;  
					//   double val1 = 1.0 - val;
					//   utMesh += log2(val1) * val1;
					if(abs(utMesh) > maxUt) {
						maxUt = abs(utMesh);
					}
					entrps.push_back(abs(utMesh));
				}
				vector<double> resEntropies;
				for (vector<double>::iterator entr = entrps.begin() ; entr != entrps.end(); ++entr) {
					double val = maxUt - *entr;
					resEntropies.push_back(val);
				}
				return resEntropies;
			}

			int predictWithEntropy(string csvFile,vector<string> annotations,string acLabel) {
				int cPred = 0;
				string cmd = "cat " + csvFile;
				int ret = system(cmd.c_str());
				fstream fs(csvFile,fstream::in);
				string line;
				getline(fs,line,'\0');
				vector<string> labels ;
				string bck_line = line;
				vector <string> testCls = split_string(line,"\n");

				for(unsigned inst = 1;inst < testCls.size(); inst++) {
					string trainfileData = testCls.at(inst);
					if(trainfileData != "") {
						vector<string> values = split_string(trainfileData,",");
						labels.push_back(values.at(0));
					}
				}
				string testNos = testCls.at(0);
				vector<string> tests = split_string(testNos,",");
				for(unsigned inst1 = 1;inst1 < tests.size(); inst1++) {
					string test_n = tests.at(inst1);
					if(test_n != "") {
						vector<double> probSamples;
						testCls = split_string(bck_line,"\n");
						for(unsigned inst = 1;inst < testCls.size(); inst++) {
							string trainfileData = testCls.at(inst);
							if(trainfileData != "") {
								vector<string> values = split_string(trainfileData,",");
								double prob = 0.0;
								if(values.size() > 2) {
									string val = values.at(inst1);
									prob = atof(val.c_str());
								}
								probSamples.push_back(prob);
							}
						}
						vector<double> entrps = calculateEntropy(probSamples);
						double mx = entrps.at(0);
						int indx = 0;
						int cIn = 0;
						for (vector<double>::iterator entr = entrps.begin() ; entr != entrps.end(); ++entr) {
							if(mx <= *entr) {
								mx = *entr;
								indx = cIn;
							}
							cIn += 1;
						}
						//cout << "Max is " << mx << " at " << indx << " Label " << labels.at(indx) << endl;
						string sample = testFullAnnotation[acLabel];
						//cout << sample << endl;
						vector<string> sampleLabels = split_string(sample," ");
						for(unsigned inst2 = 0;inst2 < sampleLabels.size(); inst2++) {
							//   cout << sampleLabels.at(inst2) << endl;
						}
						if (find(sampleLabels.begin(), sampleLabels.end(), labels.at(indx)) != sampleLabels.end()) {
							cPred += 1;
							// cout << "found " << endl;
						}

					}
				}
				//cout << " prediction is " << cPred << endl;
				return cPred;
			}

			vector<int> testTrainFiles(string objImgLoc,vector<string> nextTrainfiles,vector<string> annotations,string trainLocation,string libLinearLoc,bool syn) {
				vector<int> predEntropies;
				if(labelIndMat.size() < 3) {
					return predEntropies;
				}
				learn(trainLocation,syn);
				vector<string> folders = getFolders(trainLocation);
				for(vector<string>::iterator itan = folders.begin(); itan != folders.end(); ++itan) {
					string lbl ;
					lbl = *itan;
					string newLabelLoc = trainLocation + "/" + lbl + "/test/";
					string cmd = "rm -rf " + newLabelLoc + ";mkdir -p " + newLabelLoc;
					int ret = system(cmd.c_str());

					int fldNo = 1;
					fldNo = copyTestFiles(objImgLoc,newLabelLoc,fldNo);
				}
				string rgbCSVFile = executeModel(trainLocation,libLinearLoc) ;
				string shapeCSVFile = trainLocation + "/shapeConfusionMatrix.csv";
				string objectCSVFile = trainLocation + "/objectConfusionMatrix.csv";
				vector<string> acLabel = split_string(objImgLoc,"/");
				testWithTestFolder(shapeCSVFile,objectCSVFile,trainLocation,"testing") ;
				int prd = predictWithEntropy(rgbCSVFile,annotations,acLabel.at(acLabel.size() - 1));
				predEntropies.push_back(prd);
				prd = predictWithEntropy(shapeCSVFile,annotations,acLabel.at(acLabel.size() - 1));
				predEntropies.push_back(prd);
				prd = predictWithEntropy(objectCSVFile,annotations,acLabel.at(acLabel.size() - 1));
				predEntropies.push_back(prd);

				return predEntropies;
			}


			int verifyPrediction(string csvFile) {
				int correctPred = 0;
				cout << csvFile << endl;
				//cout << testAnnotation.size() << endl;
				fstream fs(csvFile,fstream::in);
				string line;
				getline(fs,line,'\0');
				vector<string> labels ;
				string bck_line = line;
				vector <string> testCls = split_string(line,"\n");

				for(unsigned inst = 1;inst < testCls.size(); inst++) {
					string trainfileData = testCls.at(inst);
					if(trainfileData != "") {
						vector<string> values = split_string(trainfileData,",");
						labels.push_back(values.at(0));
						//cout << values.at(0) << endl;
					}
				}
				string testNos = testCls.at(0);
				vector<string> tests = split_string(testNos,",");
				for(unsigned inst1 = 1;inst1 < tests.size(); inst1++) {
					string test_n = tests.at(inst1);
					cout << "test_n " << test_n <<  endl;
					if(test_n != "") {
						unsigned index = 0;
						double probs = 0.0;
						testCls = split_string(bck_line,"\n");
						vector<int> maxIndexes;
						for(unsigned inst = 1;inst < testCls.size(); inst++) {
							string trainfileData = testCls.at(inst);
							if(trainfileData != "") {
								vector<string> values = split_string(trainfileData,",");
								if(values.size() > 2) {
									string val = values.at(inst1);
									double prob = atof(val.c_str());
									if(prob > probs) {
										maxIndexes.clear();
										probs = prob;
										index = inst;
										maxIndexes.push_back(index);
									} else if(prob == probs) {
										maxIndexes.push_back(inst);
									}
								}
							}
						}


						string::iterator end_pos = remove(test_n.begin(), test_n.end(), ' ');
						test_n.erase(end_pos, test_n.end());

						string sample = testAnnotation[test_n];
						string mchRes = "echo '" + csvFile + " --> " + sample + " predicted as " ;
						for(unsigned inst2 = 0;inst2 < maxIndexes.size(); inst2++) {
							mchRes += " " + labels.at(maxIndexes.at(inst2) - 1);
						}
						mchRes += " ' >> ConfMatrixResults-1100-Manual/matchedResultsPredictedLabels.txt";
						int ret = system(mchRes.c_str());
						//cout << sample << endl;
						vector<string> sampleLabels = split_string(sample," ");
						for(unsigned inst2 = 0;inst2 < sampleLabels.size(); inst2++) {
							//     cout << sampleLabels.at(inst2) << endl;
						}
						if (find(sampleLabels.begin(), sampleLabels.end(), labels.at(index - 1)) != sampleLabels.end()) {
							correctPred += 1;
						}
					}
				}
				cout << "No of correct predictions " << correctPred << endl;
				return correctPred;
			}

			int verifyFullPrediction(string csvFile,int type) {
				int correctPred = 0;
				//cout << testAnnotation.size() << endl;
				fstream fs(csvFile,fstream::in);
				string line;
				getline(fs,line,'\0');
				vector<string> labels ;
				string bck_line = line;
				vector <string> testCls = split_string(line,"\n");

				for(unsigned inst = 1;inst < testCls.size(); inst++) {
					string trainfileData = testCls.at(inst);
					if(trainfileData != "") {
						vector<string> values = split_string(trainfileData,",");
						labels.push_back(values.at(0));
					}
				}
				string testNos = testCls.at(0);
				vector<string> tests = split_string(testNos,",");
				unsigned inst1 = 0;
				int tTP = 0;
				int tFP = 0;
				int tFN = 0;
				for(vector<string>::iterator it = tests.begin(); it != tests.end(); it++,inst1++ )    {
					//string test_n = tests.at(inst1);
					string test_n = *it;
					if(inst1 != 0 and test_n != "") {
						string test_n = *it;
						unsigned index = 0;
						double probs = 0.49;
						testCls = split_string(bck_line,"\n");
						vector<int> maxIndexes;
						for(unsigned inst = 1;inst < testCls.size(); inst++) {
							string trainfileData = testCls.at(inst);
							if(trainfileData != "") {
								vector<string> values = split_string(trainfileData,",");
								if(values.size() > 2) {
									string val = values.at(inst1);
									double prob = atof(val.c_str());
									if(prob > probs) {
										maxIndexes.push_back(inst - 1);
									}
								}
							}
						}


						string::iterator end_pos = remove(test_n.begin(), test_n.end(), ' ');
						test_n.erase(end_pos, test_n.end());
						string sample = "";
						if (type == 1) {
							sample = colorAnnotation[test_n];
						} else if (type == 2) {
							sample = shapeAnnotation[test_n];
						} else if (type == 3) {
							sample = objectAnnotation[test_n];
						}
						vector<string> sampleLabels = split_string(sample," ");
						for(unsigned inst2 = 0;inst2 < sampleLabels.size(); inst2++) {
							//     cout << sampleLabels.at(inst2) << endl;
						}
						vector<string> truePos;
						vector<string> falsePos;
						vector<string>  falseNeg;
						bool found = false;
						vector<string> neginstWords;
						for(unsigned inst2 = 0;inst2 < maxIndexes.size(); inst2++) {
							string lbl = labels.at(maxIndexes.at(inst2));
							end_pos = remove(lbl.begin(), lbl.end(), ' ');
							lbl.erase(end_pos, lbl.end());
							bool checkNegWord = false;
							for(unsigned inst1 = 0;inst1 < sampleLabels.size(); inst1++) {
								string ss = sampleLabels.at(inst1);
								end_pos = remove(ss.begin(), ss.end(), ' ');
								ss.erase(end_pos, ss.end());
								if (sampleLabels.at(inst1) == lbl) {
									found = true;
									checkNegWord = true;
									truePos.push_back(lbl);
								}
							}
                                                                //add to another map 
							if(checkNegWord == false) {
								neginstWords.push_back(lbl);
								falsePos.push_back(lbl);
							}
						}
						if(found){
							correctPred += 1;
						}

						for(unsigned inst1 = 0;inst1 < sampleLabels.size(); inst1++) {
							string ss = sampleLabels.at(inst1);
							end_pos = remove(ss.begin(), ss.end(), ' ');
							ss.erase(end_pos, ss.end());
							bool check = false;
							for(unsigned inst2 = 0;inst2 < maxIndexes.size(); inst2++) {
								string lbl = labels.at(maxIndexes.at(inst2));
								end_pos = remove(lbl.begin(), lbl.end(), ' ');
								lbl.erase(end_pos, lbl.end());
								if(ss == lbl) {
									check = true;
								}
							}
							if(check == false) {
								falseNeg.push_back(ss);
							}
						}
	
						int tP = 0;
						int fP = 0;
						int fN = 0;
						if(! truePos.empty())
							tP = truePos.size();
						if(! falsePos.empty()) 
							fP = falsePos.size();
						if(! falseNeg.empty()) 
							fN = falseNeg.size(); 
						tTP += tP;
						tFP += fP;
						tFN += fN;
					//	cout << "TP " << tP << " FP " << fP << " FN " << fN << endl;

						float prec = float(tP) /float(tP + fP);
						float recall = float(tP) / float(tP + fN);
					//	cout << " PRECISION " << prec << ", RECALL " << recall << endl;
				}
				}
				float tprec = float(tTP) /float(tTP + tFP);
				float trecall = float(tTP) / float(tTP + tFN);
				cout << "OVERALL - PRECISION " << tprec << ", RECALL " << trecall << endl;
				cout << "No of correct predictions " << correctPred << endl;
				return correctPred;
			}

			void testWithTestFoldersAndPredict(string dsLoc,string testConfFile,string trainLocation,string libLinearLoc,bool syn,int interval,string type) {
				learn(trainLocation,syn);
				parseAndCopySingleFile(dsLoc,testConfFile,trainLocation);
				string rgbCSVFile = executeModel(trainLocation,libLinearLoc) ;

				string shapeCSVFile = trainLocation + "/shapeConfusionMatrix.csv";
				string objectCSVFile = trainLocation + "/objectConfusionMatrix.csv";
				string cmd =  ">"+shapeCSVFile;
				int ret = system(cmd.c_str());
				cmd =  ">" + objectCSVFile;
				ret = system(cmd.c_str());
				testWithTestFolder(shapeCSVFile,objectCSVFile,trainLocation,"testing") ;

				string confMatrixLoc = "ConfMatrixResults-2100-Manual/";
				ostringstream dte;
				dte << confMatrixLoc << interval << "-" << type << "-";
				cmd =  "mkdir -p " + confMatrixLoc + ";cp " + rgbCSVFile + " " + dte.str() + "colorConfusionMatrix.csv";
				ret = system(cmd.c_str());
				cmd = "cp " + shapeCSVFile + " " + dte.str() + "shapeConfusionMatrix.csv";   
				ret = system(cmd.c_str());
				cmd = "cp " + objectCSVFile + " " + dte.str() + "objectConfusionMatrix.csv";
				ret = system(cmd.c_str());
				int cPred = verifyPrediction(rgbCSVFile); 
				ostringstream dt;
				dt << "echo \"" << interval << "-" << type << "-" << "RGB -> " << cPred << "\" >> " << confMatrixLoc <<  "PredictionResults.log";
				cmd = dt.str();
				ret = system(cmd.c_str());
				cPred = verifyPrediction(shapeCSVFile);
				ostringstream dt1;
				dt1 << "echo \"" << interval << "-" << type << "-" << "Shape -> " << cPred << "\" >> " << confMatrixLoc <<  "PredictionResults.log";
				cmd = dt1.str();
				ret = system(cmd.c_str());
				ostringstream dt2;
				dt2 << "echo \"" << interval << "-" << type << "-" << "Object -> " << cPred << "\" >> " << confMatrixLoc <<  "PredictionResults.log";
				cmd = dt2.str();
				cPred = verifyPrediction(objectCSVFile);
				ret = system(cmd.c_str());

			}

			void prepAnnotation(string testAnnot,string fullAnnot) {
				fstream fs(testAnnot,fstream::in);
				string line;
				getline(fs,line,'\0');
				vector <string> testCls = split_string(line,"\n");
				for(unsigned trainInstNo = 0;trainInstNo < testCls.size(); trainInstNo++) {
					string trainfileData = testCls.at(trainInstNo);
					if(trainfileData != "") {
						vector <string> feature = split_string(trainfileData," ");
						testAnnotation[feature.front()];
						testAnnotation[feature.front()] = trainfileData;
						//      cout << feature.front() << "--" << trainfileData << "--here s the testannotation label " << endl;
					}
				}
				fstream fs1(fullAnnot,fstream::in);
				string line1;
				getline(fs1,line1,'\0');
				testCls = split_string(line1,"\n");
				for(unsigned trainInstNo = 0;trainInstNo < testCls.size(); trainInstNo++) {
					string trainfileData = testCls.at(trainInstNo);
					if(trainfileData != "") {
						vector <string> feature = split_string(trainfileData,",");
						testFullAnnotation[feature.front()];
						testFullAnnotation[feature.front()] = feature.back();

					}

				}
				fstream fs2(colorAnnot,fstream::in);
				string line2;
				getline(fs2,line2,'\0');
				testCls = split_string(line2,"\n");
				for(unsigned trainInstNo = 0;trainInstNo < testCls.size(); trainInstNo++) {
					string trainfileData = testCls.at(trainInstNo);
					if(trainfileData != "") {
						vector <string> feature = split_string(trainfileData,",");
						string str  = feature.front();
						vector <string> feature1 = split_string(str,"/");
						str = feature1.back();
						std::string::iterator end_pos = std::remove(str.begin(), str.end(), ' ');
						str.erase(end_pos, str.end());
						colorAnnotation[str];
						colorAnnotation[str] = feature.back();

					}

				}
				fstream fs3(shapeAnnot,fstream::in);
				string line3;
				getline(fs3,line3,'\0');
				testCls = split_string(line3,"\n");
				for(unsigned trainInstNo = 0;trainInstNo < testCls.size(); trainInstNo++) {
					string trainfileData = testCls.at(trainInstNo);
					if(trainfileData != "") {
						vector <string> feature = split_string(trainfileData,",");
						string str  = feature.front();
						vector <string> feature1 = split_string(str,"/");
						str = feature1.back();
						std::string::iterator end_pos = std::remove(str.begin(), str.end(), ' ');
						str.erase(end_pos, str.end());
						shapeAnnotation[str];
						shapeAnnotation[str] = feature.back();
					}

				}
				fstream fs4(objectAnnot,fstream::in);
				string line4;
				getline(fs4,line4,'\0');
				testCls = split_string(line4,"\n");
				for(unsigned trainInstNo = 0;trainInstNo < testCls.size(); trainInstNo++) {
					string trainfileData = testCls.at(trainInstNo);
					if(trainfileData != "") {
						vector <string> feature = split_string(trainfileData,",");
						string str  = feature.front();
						vector <string> feature1 = split_string(str,"/");
						str = feature1.back();
						std::string::iterator end_pos = std::remove(str.begin(), str.end(), ' ');
						str.erase(end_pos, str.end());
						objectAnnotation[str];
						objectAnnotation[str] = feature.back();
					}

				}
			}

			void alExecuteModel(string dsLoc,string trainLocation,string trainConfFile,string testConfFile,string libLinearLoc,bool syn) {
				testAnnotation = {};
				prepAnnotation(testAnnot,fullAnnot);
				int interval = 10;
				int rgbCount = 0;
				int shapeCount = 0;
				int objectCount = 0;
				fstream fs(trainConfFile,fstream::in);
				string line;
				getline(fs,line,'\0');
				vector <string> trainCls = split_string(line,"\n");
				for(unsigned trainInstNo = 0;trainInstNo < trainCls.size(); trainInstNo++) {
					string trainfileData = trainCls.at(trainInstNo);
					if(trainfileData != "") {
						cout << trainfileData << endl;
						vector <string> feature = split_string(trainfileData,",");
						string objImgLoc = dsLoc + "/" + feature.front();
						vector<string> nextTrainfiles = getSpecFolders(objImgLoc);
						string annotation = feature.back();
						vector<string> anntn = processLanguage(annotation);
						vector<int> predEntropies = testTrainFiles(objImgLoc,nextTrainfiles,anntn,trainLocation,libLinearLoc,syn);
						if(predEntropies.size() == 0) {
							rgbCount += 1; shapeCount += 1; objectCount += 1;
						} else if(predEntropies.at(0)  < 1) {
							rgbCount += 1;
						} else if(predEntropies.at(1)  < 1) {
							shapeCount += 1;
						} else if(predEntropies.at(2)  < 1) {
							objectCount += 1;
						}

						addtrainFolderUnderAnnotation(objImgLoc,trainLocation,anntn,nextTrainfiles,true,syn);
						cout << "Current counts ---->" << endl;
						cout << "Manual " << trainInstNo + 1 << ", RGB " << rgbCount << ",Shape " << shapeCount << ",Object " << objectCount<< endl; 
						int count;
						string key;
						bool flag = false;
						if(((trainInstNo + 1) % interval) == 0) {
							count = trainInstNo + 1;
							key = "manual";
							flag = true;
						} else if((rgbCount % interval) == 0) {
							count = rgbCount;
							key = "al-rgb";
							flag = true;
						} else if ((shapeCount % interval) == 0) {
							count = shapeCount;
							key = "al-shape";
							flag = true;
						} else if ((objectCount % interval) == 0) {
							count = objectCount;
							key = "al-object";
							flag = true;
						}
						if(flag) {
							cout << "Train Instance Number :: " << count  << " - " << key << endl;
							testWithTestFoldersAndPredict(dsLoc,testConfFile,trainLocation,libLinearLoc,syn,(trainInstNo + 1),"manual");

						}  
					}
				}
			}
			vector<string> fileToVector(string trainConfFile) {
				fstream fs(trainConfFile,fstream::in);
				string line;
				getline(fs,line,'\0');
				vector<string> trainCls;
				vector <string> trainCls1 = split_string(line,"\n");
				for(unsigned trainInstNo = 0;trainInstNo < trainCls1.size(); trainInstNo++) {
					string trainfileData = trainCls1.at(trainInstNo);
					if(trainfileData != "") {
						trainCls.push_back(trainfileData);
					}
				}
				return trainCls;

			}

			vector<string> randomTrainAddition(string dsLoc,vector<string> trainCls,int index,string trainLocation,bool syn) {
				string trainfileData = trainCls.at(index);
				trainCls.erase(trainCls.begin() + index);
				cout << trainfileData << endl;
				vector <string> feature = split_string(trainfileData,",");
				string objImgLoc = dsLoc + "/" + feature.front();
				vector<string> nextTrainfiles = getSpecFolders(objImgLoc);
				string annotation = feature.back();
				//   vector<string> anntn1 = processLanguage(annotation);
				vector<string> anntn1 = split_string(annotation," ");
				vector<string> anntn;
				for(unsigned i = 0;i < anntn1.size(); i++) {
					if(anntn1.at(i) != "") {
						anntn.push_back(anntn1.at(i));
					}
				}
				if (anntn.size() > 0) {
					addtrainFolderUnderAnnotation(objImgLoc,trainLocation,anntn,nextTrainfiles,true,syn);
				}
				return trainCls;


			}

			void   manualExecutionAndPrediction(string dsLoc,string trainLocation,string trainConfFile,string testConfFile,string libLinearLoc,bool syn) {
				int interval = 25;

				testAnnotation = {};
				prepAnnotation(testAnnot,fullAnnot);

				vector<string> trainCls = fileToVector(trainConfFile);
				interval = trainCls.size();
				int count = 0;
				while(trainCls.size() > 0) {
					for(unsigned i = 0; i < interval;i++) {
						if(trainCls.size() > 0) {
							srand(time(NULL));
							count = count + 1;
							int index = rand() % trainCls.size();
							trainCls = randomTrainAddition(dsLoc,trainCls,index,trainLocation,syn);
						}
					}
					cout << "Train Instance Number :: " << count  << endl;
					testWithTestFoldersAndPredict(dsLoc,testConfFile,trainLocation,libLinearLoc,syn,count,"manual");
				}

			}

			void onlineManualtrain(string croppedFiles,string speechLog) {


			}
			void getLanguage(string spLoc,string spAudioLoc) {
				voiceOut("Please explain me about the object",true);
				string cmd = "sh " + spLoc + " " + spAudioLoc;
				cout << cmd << endl;
				int ret = system(cmd.c_str());
			}

			vector<double> findEntropy(vector<double> testProbs) {
				vector<double> testEntrs;
				for(unsigned i = 0; i < testProbs.size();i++) {
					double val = testProbs.at(i);
 					
					double entr =  log2(val) * val ;
 
					testEntrs.push_back(entr); 
				}
				return testEntrs;
			}

			vector<double> findEntropyTotal(vector<double> testProbs) {
                                vector<double>  entrIndices;
                                for(unsigned i = 0; i < testProbs.size();i++) {
                                        double val = testProbs.at(i);
                                        double negVal = 1.0 - val;
                                        double entr = (-1 * log2(val) * val );
					double entr1 =  entr + (-1 * log2(negVal) * negVal);
					entrIndices.push_back(entr1);
				}
                                return entrIndices;
			}

			vector<int> findEntropyAL(vector<double> testProbs, float thresh) {
				vector<int>  entrIndices;
				for(unsigned i = 0; i < testProbs.size();i++) {
					double val = testProbs.at(i);
 					double negVal = 1.0 - val;
					double entr = (-1 * log2(val) * val );
                                         double entr1 =  entr + (-1 * log2(negVal) * negVal);
					if(entr1 / val < thresh) {
						entrIndices.push_back(i);
                                      
					}
				}
				return entrIndices;
			}
			vector<int> findPointsToBeLabelled(string csvFile) {
				string cmd = "cat " + csvFile;
				int ret = system(cmd.c_str());
				fstream fs(csvFile,fstream::in);
				string line;
				getline(fs,line,'\0');
				vector<string> labels ;
				string bck_line = line;
				vector <string> testCls = split_string(line,"\n");
				for(unsigned inst = 1;inst < testCls.size(); inst++) {
					string trainfileData = testCls.at(inst);
					if(trainfileData != "") {
						vector<string> values = split_string(trainfileData,",");
						labels.push_back(values.at(0));
					}
				}
				string testNos = testCls.at(0);
				vector<double> probMatches;
				vector<string> tests = split_string(testNos,",");
				for(unsigned inst1 = 1;inst1 < tests.size(); inst1++) {
					string test_n = tests.at(inst1);
					vector<double> testProbs;
					if(test_n != "") {
						testCls = split_string(bck_line,"\n");
						for(unsigned inst = 1;inst < testCls.size(); inst++) {
							string trainfileData = testCls.at(inst);
							if(trainfileData != "") {
								vector<string> values = split_string(trainfileData,",");
								double prob = 0.0;
								if(values.size() > 2) {
									string val = values.at(inst1);
									prob = atof(val.c_str());
								}
								testProbs.push_back(prob);
							}
						}
					}
					vector<double> testEntrs = findEntropy(testProbs);    
					auto mEl = max_element(begin(testEntrs), end(testEntrs)); 
					probMatches.push_back(abs(*mEl));
				}
				priority_queue<pair<double, int>> q;
				for (int i = 0; i < probMatches.size(); ++i) {
					double prob = probMatches.at(i);
					string str = tests.at(i + 1);
					str.replace(str.begin(),str.begin()+ 5,"");
					int v = atoi(str.c_str());
					q.push(pair<double, int>(prob, v));
				}
				vector<int> indexes;
				for (int i = 0; i < probMatches.size(); ++i) {
					int ki = q.top().second;
					indexes.push_back(ki);
					q.pop();
				}
				cout << "test_" << indexes.at(0) << endl;
				return indexes;
			}

			vector<int> predUnCertainity(string csvFile) {
				string cmd = "cat " + csvFile;
				int ret = system(cmd.c_str());
				fstream fs(csvFile,fstream::in);
				string line;
				getline(fs,line,'\0');
				vector<string> labels ;
				string bck_line = line;
				vector <string> testCls = split_string(line,"\n");
				for(unsigned inst = 1;inst < testCls.size(); inst++) {
					string trainfileData = testCls.at(inst);
					if(trainfileData != "") {
						vector<string> values = split_string(trainfileData,",");
						labels.push_back(values.at(0));
					}
				}
				string testNos = testCls.at(0);
				vector<int> probMatches;
				vector<string> tests = split_string(testNos,",");
				for(unsigned inst1 = 1;inst1 < tests.size(); inst1++) {
					string test_n = tests.at(inst1);
					vector<double> testProbs;
					double maxProb = 0.0;
					int indexMatch = 0;
					if(test_n != "") {
						testCls = split_string(bck_line,"\n");
						for(unsigned inst = 1;inst < testCls.size(); inst++) {
							string trainfileData = testCls.at(inst);
							if(trainfileData != "") {
								vector<string> values = split_string(trainfileData,",");
								double prob = 0.0;
								if(values.size() > 2) {
									string val = values.at(inst1);
									prob = atof(val.c_str());
									if(prob >= maxProb) {
										maxProb = prob;
										indexMatch = inst - 1;
									}
								}
								testProbs.push_back(prob);
							}
						}

					}
					cout << test_n << "---" << maxProb << "---" << labels.at(indexMatch) << "==" << indexMatch << endl;
					//   probMatches.push_back(match);
				}
				return probMatches;
			}

			vector<int> onlineAltrain(string objImgLoc,string trainLocation,string libLinearLoc) {
				//    vector<string> folders = getFolders(trainLocation);
				//   for(vector<string>::iterator itan = folders.begin(); itan != folders.end(); ++itan) {
				//       string lbl ;
				//       lbl = *itan;
				//       string newLabelLoc = trainLocation + "/" + lbl + "/test/";
				//       string cmd = "rm -rf " + newLabelLoc + ";mkdir -p " + newLabelLoc;
				//      int ret = system(cmd.c_str());

				//       int fldNo = 1;
				//       fldNo = copyTestFiles(objImgLoc,newLabelLoc,fldNo);
				//   }
				//   string rgbCSVFile = executeModel(trainLocation,libLinearLoc) ;
				string shapeCSVFile = trainLocation + "/shapeConfusionMatrix.csv";
				string objectCSVFile = trainLocation + "/objectConfusionMatrix.csv";
				testWithTestFolder(shapeCSVFile,objectCSVFile,trainLocation,"altesting") ;
				vector<int> resTrain;
				vector<int> prd2 = predUnCertainity(objectCSVFile);
				//   for(unsigned inst1 = 0;inst1 < prd.size(); inst1++) {
				//     int res = prd2.at(inst1);
				//     resTrain.push_back(res);
				//  }
				return resTrain;
			}

			vector<int> copyTestFromFolder(string dsLoc,string trainLocation,vector<string> cls) {
				vector<int> fileIndexes; 
				vector<string> folders = getFolders(trainLocation);
				cout << "Adding Test folders " << endl;
				string tempTestLoc = dsLoc + "/testtest";
				fileIndexes = prepareTestFolders(dsLoc,cls,tempTestLoc);
				for(vector<string>::iterator itan = folders.begin(); itan != folders.end(); ++itan) {
					string lbl ;
					lbl = *itan;
					string newLabelLoc = trainLocation + "/" + lbl + "/test";
					string cmd = "rm -rf " + newLabelLoc + ";";
					cmd += "cp -r " + tempTestLoc + " " +newLabelLoc;
					int ret = system(cmd.c_str());
				}
				cout << "Done :: Test examples added " << endl;
				return fileIndexes;
			}

			vector<int> alSelectUnlabelledPoints(string dsLoc,string trainLocation,string libLinearLoc,vector<string> trainCls,int alInterval,bool syn) {

				set<string> unlabelledFolders;
				for (unsigned index = 0; index < trainCls.size();index++) {
					string trainfileData = trainCls.at(index);
					vector <string> feature = split_string(trainfileData,",");
					unlabelledFolders.insert(feature.front());
				}
				vector<string> unlabelledV;
				copy(unlabelledFolders.begin(), unlabelledFolders.end(), inserter(unlabelledV,unlabelledV.begin())); 
				learn(trainLocation,syn);
				vector<int> fileIndexes = copyTestFromFolder(dsLoc,trainLocation,unlabelledV);
				//   string rgbCSVFile = executeModel(trainLocation,libLinearLoc) ;
				string shapeCSVFile = trainLocation + "/shapeConfusionMatrix.csv";
				string objectCSVFile = trainLocation + "/objectConfusionMatrix.csv";
				string cmd =  ">"+shapeCSVFile;
				int ret = system(cmd.c_str());
				cmd =  ">" + objectCSVFile;
				ret = system(cmd.c_str());
				testWithTestFolder(shapeCSVFile,objectCSVFile,trainLocation,"testing") ;  
				vector<int> indexes = findPointsToBeLabelled(objectCSVFile);
				vector<int> cIndexes;
				for(unsigned  k = 0; k < indexes.size() ; k++) {
					int kj = indexes.at(k) - 1;
					cIndexes.push_back(fileIndexes.at(kj));
				}
				return cIndexes;
			}

			void activeExecEntropyAndPrediction(string dsLoc,string trainLocation,string trainConfFile,string testConfFile,string libLinearLoc,bool syn) {

				int interval = 25;
				int intialManualInterval = 10;
				testAnnotation = {};
				prepAnnotation(testAnnot,fullAnnot);

				vector<string> trainCls = fileToVector(trainConfFile);
				int alInterval;
				int count = 0;
				while(trainCls.size() > 0) {
					for(unsigned i = 0; i < intialManualInterval;i++) {
						if(trainCls.size() > 0) {
							count = count + 1;
							srand(time(NULL));
							int index = rand() % trainCls.size();
							trainCls = randomTrainAddition(dsLoc,trainCls,index,trainLocation,syn);
						}
					}

					alInterval = interval - intialManualInterval;
					vector<int> indexes = alSelectUnlabelledPoints(dsLoc,trainLocation,libLinearLoc,trainCls,alInterval,syn);
					set<int> nIndx;
					unsigned kj = 0;

					while(nIndx.size() < alInterval and kj < indexes.size()) {
						nIndx.insert(indexes.at(kj));
						kj += 1;
					}


					vector<int> ind;
					copy(nIndx.begin(), nIndx.end(), inserter(ind,ind.begin()));
					for(unsigned i = 0; i < alInterval;i++) {
						if(trainCls.size() > 0) {
							count = count + 1;
							trainCls = randomTrainAddition(dsLoc,trainCls,ind.at(i),trainLocation,syn);
						}
					}
					intialManualInterval = 0;
					cout << "Train Instance Number :: " << count  << endl;
					testWithTestFoldersAndPredict(dsLoc,testConfFile,trainLocation,libLinearLoc,syn,count,"alentropy");
				}

			}

			int prepareNegMap(string confFile) {
				vector<string> trainCls = fileToVector(confFile);
				for(unsigned i = 0; i < trainCls.size();i++) {
					string trainfileData = trainCls.at(i);
					if(trainfileData != "") {
						vector <string> feature = split_string(trainfileData,",");
						string annotation = feature.back();
						vector<string> anntn = processLanguage(annotation);
						vector<string> antn1 = anntn;
						for(vector<string>::iterator itan = anntn.begin(); itan != anntn.end(); ++itan) {
							string lbl ;
							lbl = *itan;
							negLabelMapUpdate(antn1,lbl);
						}
					}
				}
				return trainCls.size();
			}

			vector<int>  prepTestFolder(string dsLoc,string testConfFile,string tempTestLoc) {
				fstream fs(testConfFile,fstream::in);
				string line;
				getline(fs,line,'\0');
				vector <string> cls = split_string(line,"\n");
				vector<string> cls1;
				for(unsigned i = 0; i <cls.size(); i++) {
					if(cls.at(i) != "") {
						vector<string> n = split_string(cls.at(i),",");
						cls1.push_back(n.front());
					}
				}
				return prepareTestFolders(dsLoc,cls1,tempTestLoc);
			}

			void updateFolder(string lbl, vector<string> poss,vector<string> negs,string trainLocation,string tempTestLoc,int negType) {
				string newLabelLoc = trainLocation + "/" + lbl + "/"+lbl+"Pos/";
				string cmd = "mkdir -p " + newLabelLoc;
				int ret = system(cmd.c_str());
				ofstream myfile;
				myfile.open(negPosInstances,std::ios_base::app);
				myfile << lbl << " --> \nPositive Instances :: ";
				for(vector<string>::iterator itn = poss.begin(); itn != poss.end(); ++itn) {
					string img;
					img = *itn;
					myfile << img << ", ";
					vector<string> files2 = getFolders(newLabelLoc);
					int newNo = files2.size();
					int newFldr = newNo + 1;

					ostringstream dte1;
					dte1 << newLabelLoc << lbl << "Pos_" << newFldr;
					string newLabelLoc1 = dte1.str();
					renameTFilesAndCopy(img,newLabelLoc1);
				}
				myfile.close();
				if (negType == 1) {
					addTNeg(trainLocation,lbl,negs);
				} else if (negType == 2) {
					addTNegTFIDF(trainLocation,lbl,negs);
				}
				///adding test files
				newLabelLoc = trainLocation + "/" + lbl + "/test";
				cmd = "rm -rf " + newLabelLoc + ";";
				cmd += "cp -r " + tempTestLoc + " " +newLabelLoc;
				ret = system(cmd.c_str());
			}

			void testIndModel(string trainFile,string testFile, int cNo,string results) {
				ostringstream sM;
				sM << "cd  /home/npillai1/donengel_common/Nisha;";
				sM << "java -Xmx6g -cp ";
				sM << wekaLoc;
				sM << " ";
				sM << classifier;
				sM << " -t ";
				sM << trainFile;
				sM << " -T ";
				sM << testFile;
				sM << " -p 1-" << cNo;

				sM << " > ";
				sM << results;
				int ret = system(sM.str().c_str());
			}

			string arffHead(string label,string id) {
				string head = "@RELATION " + label + id + "\n\n";
				if(id == "rgb" or id == "object") {
					head += "@ATTRIBUTE R NUMERIC\n";
					head += "@ATTRIBUTE G NUMERIC\n";
					head += "@ATTRIBUTE B NUMERIC\n";
				} 
				if(id == "shape" or id == "object") {
					int i = 1;
					ostringstream dat;

					while(i <= 14000) {
						dat << "@ATTRIBUTE A" << i << " NUMERIC\n";   
						i += 1;
					}
					head += dat.str();
				}
				head += "@ATTRIBUTE class {1,-1}\n\n";
				head += "@DATA\n";

				return head;
			}

			void wekatrffCreation(string trainLoc,string label,string libLinearLoc,string csvFile,string id) {
				string trainSet = trainLoc + "/" + label + "/" + id + "-trainSet.arff";
				string testSet = trainLoc + "/" + label + "/" + id + "-testSet.arff";
				string str = "rm -rf "+ trainSet;
				int ret = system(str.c_str());
				str = "rm -rf "+ testSet;
				ret = system(str.c_str());
				string idFile = "_" + id + ".log";
				string head = arffHead(label,id);
				string imNm = head;
				string classLabel = " 1";
				string clsName = label + "Pos";
				vector<string> clsFolders = getFolders(trainLoc + "/" + label + "/" + clsName + "/");
				for(vector<string>::iterator itClssub = clsFolders.begin(); itClssub != clsFolders.end(); ++itClssub) {
					string clsInstName = *itClssub;
					string imgName = trainLoc + "/" + label + "/" + clsName + "/" + clsInstName + "/" + clsInstName + idFile;
					ifstream file(imgName);
					string str;
					while (getline(file, str))
					{
						if (str != "") {
							vector <string> elements = split_string(str," "); 

							for(vector<string>::iterator itElem = elements.begin(); itElem != elements.end(); ++itElem) {
								string elem = *itElem;
								if (elem != "" ) {
									vector <string> elems2 = split_string(elem,":");    
									string elem2 = elems2.at(1);
									imNm += elem2 + ",";
								}
							}
							imNm += classLabel + "\n";
						}
					}
				}
				ofstream ofs;
				ofs.open (trainSet, ofstream::out);
				ofs << imNm;
				ofs.close();


				imNm = "";
				classLabel = "-1 ";
				clsName = label + "Neg";
				clsFolders = getFolders(trainLoc + "/" + label + "/" + clsName + "/");
				for(vector<string>::iterator itClssub = clsFolders.begin(); itClssub != clsFolders.end(); ++itClssub) {
					string clsInstName = *itClssub;
					string imgName = trainLoc + "/" + label + "/" + clsName + "/" + clsInstName + "/" + clsInstName + idFile;
					ifstream file(imgName);
					string str;
					while (getline(file, str))
					{  
						if (str != "") {
							vector <string> elements = split_string(str," ");

							for(vector<string>::iterator itElem = elements.begin(); itElem != elements.end(); ++itElem) {
								string elem = *itElem;
								if (elem != "" ) {
									vector <string> elems2 = split_string(elem,":");
									string elem2 = elems2.at(1);
									imNm += elem2 + ",";
								}
							}
							imNm += classLabel + "\n";
						}
					}
				}
				ofs.open (trainSet, ofstream::app);
				ofs << imNm;
				ofs.close();

				imNm = "";
				imNm = head;
				classLabel = "1 ";
				clsName = "test";
				clsFolders = getFolders(trainLoc + "/" + label + "/" + clsName + "/");
				for (int i = 1; i <= clsFolders.size(); i++) {
					ostringstream dat;
					dat << "test_" << i ;
					string clsInstName = dat.str();
					string imgName = trainLoc + "/" + label + "/" + clsName + "/" + clsInstName + "/" + clsInstName + idFile;
					ifstream file(imgName);
					string str;
					while (getline(file, str))
					{
						if (str != "") {
							vector <string> elements = split_string(str," ");

							for(vector<string>::iterator itElem = elements.begin(); itElem != elements.end(); ++itElem) {
								string elem = *itElem;
								if (elem != "" ) {
									vector <string> elems2 = split_string(elem,":");
									string elem2 = elems2.at(1);
									imNm += elem2 + ",";
								}
							}
							imNm += classLabel + "\n";
						}
					}
				}
				ofs.open (testSet, ofstream::out);
				ofs << imNm;
				ofs.close();
				int cNo = 3;
				if(id == "shape") {
					cNo = 14000;
				}
				if (id == "object") {
					cNo += 14000;
				}   
				string resultFile = trainLoc + "/" + label + "/" + id + "-" + label + "-wekaresults.log";
				//   testIndModel(trainSet,testSet, cNo,resultFile);

			}
			void prepFileLearnAndTest(string trainLoc,string label,string libLinearLoc,string csvFile,string id) {
				//wekatrffCreation(trainLoc,label,libLinearLoc,csvFile,id);
				string idFile = "_" + id + ".log";
				string tFile  = trainLoc + "/temp.csv";
				string cm = " >"+tFile;
				int ret1 = system(cm.c_str());
				string trainSet = trainLoc + "/" + label + "/" + id + "-trainSet.linear";
				string testSet = trainLoc + "/" + label + "/" + id + "-testSet.linear";
				string modelFile = trainLoc + "/" + label + "/" + label + "-" + id + "-classification.model";
				string outputFile = trainLoc + "/" + label + "/" + label + "-" + id + "-out.file";
				string str = "rm -rf "+ trainSet;
				int ret = system(str.c_str());
				str = "rm -rf "+ testSet;
				ret = system(str.c_str());

				string imNm = "";
				string classLabel = "1 ";
				string clsName = label + "Pos";
				vector<string> clsFolders = getFolders(trainLoc + "/" + label + "/" + clsName + "/");
				for(vector<string>::iterator itClssub = clsFolders.begin(); itClssub != clsFolders.end(); ++itClssub) {
					string clsInstName = *itClssub;
					string imgName = trainLoc + "/" + label + "/" + clsName + "/" + clsInstName + "/" + clsInstName + idFile;
					ifstream file(imgName);
					string str; 
					while (getline(file, str))
					{
						if (str != "") {
							imNm += classLabel + str + "\n";
						}  
					}
				}
				ofstream ofs;
				ofs.open (trainSet, ofstream::out);
				ofs << imNm;
				ofs.close();


				imNm = "";
				classLabel = "-1 ";
				clsName = label + "Neg";
				clsFolders = getFolders(trainLoc + "/" + label + "/" + clsName + "/");
				for(vector<string>::iterator itClssub = clsFolders.begin(); itClssub != clsFolders.end(); ++itClssub) {
					string clsInstName = *itClssub;
					string imgName = trainLoc + "/" + label + "/" + clsName + "/" + clsInstName + "/" + clsInstName + idFile;
					ifstream file(imgName);
					string str;
					while (getline(file, str))
					{
						if (str != "") {
							imNm += classLabel + str + "\n";
						}
					}
				}
				ofs.open (trainSet, ofstream::app);
				ofs << imNm;
				ofs.close();

				imNm = "";
				classLabel = "1 ";
				clsName = "test";
				clsFolders = getFolders(trainLoc + "/" + label + "/" + clsName + "/");
				for (int i = 1; i <= clsFolders.size(); i++) { 
					ostringstream dat;
					dat << "test_" << i ;
					string clsInstName = dat.str();
					string imgName = trainLoc + "/" + label + "/" + clsName + "/" + clsInstName + "/" + clsInstName + idFile;
					ifstream file(imgName);
					string str;
					while (getline(file, str))
					{
						if (str != "") {
							imNm += classLabel + str + "\n";
						}
					}
				}
				ofs.open (testSet, ofstream::out);
				ofs << imNm;
				ofs.close();

				string cmd1 = libLinearLoc + "/train -s 0 -c 10 " + trainSet + " " + modelFile;
				ret = system(cmd1.c_str());
				cmd1 = libLinearLoc + "/predict -b 1 " + testSet + " " + modelFile + " " + outputFile;
				ret = system(cmd1.c_str());

				fstream fs(outputFile,fstream::in);
				string line;
				getline(fs,line,'\0');
				vector <string> predictions = split_string(line,"\n");
				unsigned posIndex = 1;
				string probabilities = "";
				for (vector<string>::iterator pred = predictions.begin() ; pred != predictions.end(); ++pred) {
					string clsPred = *pred;
					string chkName = "labels";
					vector<string> probs = split_string(clsPred," ");
					if (clsPred.find(chkName) != string::npos) {
						if (probs.at(2) == "1") {
							posIndex = 2;
						}
					} else if (clsPred != "") {
						probabilities += probs.at(posIndex) + ", ";
					}
				}
				cmd1 = "echo " + label + "," + probabilities +">>" + csvFile;
				ret = system(cmd1.c_str());
				
				//testWithLogisticRegression(label,csvFile,trainSet,testSet);
			}

			void testColorShapeObject(string trainLocation,string label,string libLinearLoc,string rgbCSVFile,string shapeCSVFile,string objectCSVFile)  {
                              
				prepFileLearnAndTest(trainLocation,label,libLinearLoc,rgbCSVFile,"rgb");
				prepFileLearnAndTest(trainLocation,label,libLinearLoc,shapeCSVFile,"shape");
				prepFileLearnAndTest(trainLocation,label,libLinearLoc,objectCSVFile,"object");
				vector<string> category = {"rgb","shape","object"};
                                vector<string> csvFiles = {rgbCSVFile,shapeCSVFile,objectCSVFile};
				cout << category.size() <<endl;
				//testLogRegression(trainLocation,label,csvFiles,category);
				
			}

			vector<string> saveTestTrueValues(vector<string> testSet,string tempTestLoc,vector<string> trainCls) {
				string clsName = "test";
				vector<string> clsFolders = getFolders(tempTestLoc + "/");
				vector<string> testLabels;
				for (int i = 1; i <= clsFolders.size(); i++) {
					ostringstream dat;
					dat << "test_" << i ;
					string clsInstName = dat.str();
					string imgName = tempTestLoc + "/" + clsInstName + "/" + clsInstName + "_rgb.log";;
					ifstream file(imgName);
					string str;
					while (getline(file, str))
					{  
						if (str != "") {
							string s = testSet.at(i - 1);
							vector<string> aab = split_string(s,"/");
							// string abc = testFullAnnotation[aab.at(1)];
							// testLabels.push_back(abc);
							testLabels.push_back(aab.at(1));
							
						}
					}
				}
				return testLabels;
			}


			int addFoldersAndTestAllAL(string trainLocation,string tempTestLoc,string libLinearLoc,vector<string> testLabels,bool syn,int negType) {
				string cmd = "mkdir -p " + trainLocation;
				int ret = system(cmd.c_str());

				string rgbCSVFile = trainLocation + "/rgbConfusionMatrix.csv";
				cmd = ">"+ rgbCSVFile;
				ret = system(cmd.c_str());
				cout << "-------RGB CSV FILE--------------- " << rgbCSVFile << endl;
				string shapeCSVFile = trainLocation + "/shapeConfusionMatrix.csv";
				cmd = ">"+ shapeCSVFile;
				ret = system(cmd.c_str());
				cout << "-------Shape CSV FILE--------------- " << shapeCSVFile << endl;
				string objectCSVFile = trainLocation + "/objectConfusionMatrix.csv";
				cmd = ">"+ objectCSVFile;
				ret = system(cmd.c_str());
				cout << "-------Object CSV FILE--------------- " << objectCSVFile << endl;
				string lbls = ",";
				for (int i = 0; i < testLabels.size() ; i ++ ) {
					lbls += testLabels.at(i) + ",";
				}
				cout << "--->NASH-->" << lbls << endl;
				cmd = "echo " + lbls +">>" + rgbCSVFile;
				ret = system(cmd.c_str());
				cmd = "echo " + lbls +">>" + shapeCSVFile;
				ret = system(cmd.c_str());
				cmd = "echo " + lbls +">>" + objectCSVFile;
				ret = system(cmd.c_str());
/*
//				for(double l_rate = 0.1; l_rate <= 1.00; l_rate += 0.1) { 
//					for(int n_epoch = 100;n_epoch <10000; n_epoch += 100) {
				for(double l_rate = 0.1; l_rate <= 0.1; l_rate += 1.00) { 
					for(int n_epoch = 5000;n_epoch <= 5000; n_epoch += 10000) {
*/
					int n_epoch = 5000;
					double l_rate = 0.1;
						ostringstream dte;
						dte << rgbCSVFile << "-" << n_epoch << "-" << l_rate << ".csv";
						string csF = dte.str();
						cmd = "echo " + lbls +">>" + csF;
						ret = system(cmd.c_str());

                                                ostringstream dte1;
                                                dte1 << shapeCSVFile << "-" << n_epoch << "-" << l_rate << ".csv";
                                                csF = dte1.str();
                                                cmd = "echo " + lbls +">>" + csF;
                                                ret = system(cmd.c_str());

                                                ostringstream dte2;
                                                dte2 << objectCSVFile << "-" << n_epoch << "-" << l_rate << ".csv";
                                                csF = dte2.str();
                                                cmd = "echo " + lbls +">>" + csF;
                                                ret = system(cmd.c_str());

//					}
//				} 

				cout << "All Words -> ";
				for(map<string,vector<string>>::iterator itit=posLabelIndMat.begin();itit != posLabelIndMat.end();++itit) {
					ostringstream dat;
					dat << itit->first;
					string lbl = dat.str();
					cout << lbl << ", ";
				}
				cout << endl << endl;
				ofstream myfile;
				myfile.open(negPosInstances);
				myfile << endl;
				myfile.close();
				for(map<string,vector<string>>::iterator itit=posLabelIndMat.begin();itit != posLabelIndMat.end();++itit) {
					ostringstream dat;
					dat << itit->first;
					string lbl = dat.str();
					cout << lbl << " is starting execution !!!!!" << endl;
//					if (lbl == "arcshape") {
//					if (lbl == "apple" or lbl == "arch" or lbl == "archshaped" or lbl == "arcshape" or lbl == "banana" or lbl == "black" or lbl == "block" or lbl == "blue" ){
					cout << endl << " >>>>>>  " << lbl << "--->";
					vector<string> poss = posLabelIndMat[lbl];
					vector<string> negss = labelIndMat[lbl];
					set<string> snegs(negss.begin(), negss.end());
					vector<string> negs(snegs.begin(), snegs.end());
					cout <<  "Pos # " << poss.size() << ", Neg # " << negs.size() << endl;
					if (poss.size() > minInstances and negs.size() > minInstances) {
						updateFolder(lbl,poss,negs,trainLocation,tempTestLoc,negType);
						testColorShapeObject(trainLocation,lbl,libLinearLoc,rgbCSVFile,shapeCSVFile,objectCSVFile) ;
						// if (lbl != "carrot") {           
						string newLabelLoc = trainLocation + "/" + lbl ;      
						cmd = "rm -rf " + newLabelLoc;
						ret = system(cmd.c_str());
						string rmFiles = trainLocation + "/" + lbl + "/*.linear";
						cmd = "rm -rf " + rmFiles;
						ret = system(cmd.c_str());
						rmFiles = trainLocation + "/" + lbl + "/test";
						cmd = "rm -rf " + rmFiles;
						ret = system(cmd.c_str());
						rmFiles = trainLocation + "/" + lbl + "/" + lbl + "*";           
						cmd = "rm -rf " + rmFiles;
						ret = system(cmd.c_str()); 
						//  }  
					}
					cout << endl << lbl << " - Completed execution !!!!!" << endl;
//				}

				}
/*
                                for(double l_rate = 0.1; l_rate <= 0.1; l_rate += 1.05) {
                                        for(int n_epoch = 5000;n_epoch <= 5000; n_epoch += 1000) {
*/
				n_epoch = 5000;
				l_rate = 0.1;
                                                ostringstream dt1;
                                                dt1 << rgbCSVFile << "-" << n_epoch << "-" << l_rate << ".csv";
                                                csF = dt1.str();
						cout << csF << endl;
						int predCount = verifyFullPrediction(csF,1);

                                                ostringstream dt2;
                                                dt2 << shapeCSVFile << "-" << n_epoch << "-" << l_rate << ".csv";
                                                csF = dt2.str();
                                                cout << csF << endl;
                                                predCount = verifyFullPrediction(csF,1);

                                                ostringstream dt3;
                                                dt3 << objectCSVFile << "-" << n_epoch << "-" << l_rate << ".csv";
                                                csF = dt3.str();
                                                cout << csF << endl;
                                                predCount = verifyFullPrediction(csF,1);


/*					}
				} 
*/
				predCount = 	verifyFullPrediction(rgbCSVFile,1);
/*
				int val = verifyFullPrediction(shapeCSVFile,2);
				if ( predCount > val) {
					predCount = val;
				}
				val = verifyFullPrediction(objectCSVFile,3);
				if ( predCount > val) {
					predCount = val;
				}
*/
				if(syn) {
					vector<string> labelKeys;
					for(map<string,vector<string>>::iterator itit=posLabelIndMat.begin();itit != posLabelIndMat.end();++itit) {
						ostringstream dat;
						dat << itit->first;
						labelKeys.push_back(dat.str());
					}
					for(map<string,vector<string>>::iterator itit=posLabelIndMat.begin();itit != posLabelIndMat.end();++itit) {
						ostringstream dat;
						dat << itit->first;
						string lbl = dat.str();
						vector<string> clblKeys;
						int index = 0;
						bool found = false;
						while (not found) {
							string abc = labelKeys.at(index);
							if (abc == lbl) {
								found = true;
							} else {
								clblKeys.push_back(abc);
								index += 1;
							}
						}
						for (int i = 0; i < clblKeys.size() ; i ++ ) {
							string extrLbl = clblKeys.at(i);
							vector<string> poss = posLabelIndMat[lbl];
							vector<string> extrPoss = posLabelIndMat[extrLbl];
							copy(extrPoss.begin(), extrPoss.end(), back_inserter(poss)); 

							vector<string> lblNegs = labelIndMat[lbl];
							vector<string> extrNegs = labelIndMat[extrLbl];

							vector<string> negs;

							sort(lblNegs.begin(), lblNegs.end());
							sort(extrNegs.begin(), extrNegs.end());

							set_intersection(lblNegs.begin(),lblNegs.end(),extrNegs.begin(),extrNegs.end(),back_inserter(negs));
							if (poss.size() > minInstances and negs.size() > minInstances) {
								string synLbl = lbl + "-" + extrLbl;
								cout << synLbl << endl;
								updateFolder(synLbl,poss,negs,trainLocation,tempTestLoc,negType);
								testColorShapeObject(trainLocation,synLbl,libLinearLoc,rgbCSVFile,shapeCSVFile,objectCSVFile) ;

								string newLabelLoc = trainLocation + "/" + synLbl ;
								cmd = "rm -rf " + newLabelLoc;
								ret = system(cmd.c_str());
							}
						}

					}
				} 
				return predCount;
			}

			void addFoldersAndTestAll(string trainLocation,string tempTestLoc,string libLinearLoc,vector<string> testLabels,bool syn,int negType) {
				int val = addFoldersAndTestAllAL(trainLocation,tempTestLoc,libLinearLoc,testLabels,syn,negType);
			}

			vector<string> prepareTestData(vector<string> trainCls) {
				set<string> fullDataSet;
				for(unsigned i = 0;i < trainCls.size(); i++) {
					if(trainCls.at(i) != "") {
						string trainfileData = trainCls.at(i);
						vector <string> feature = split_string(trainfileData,",");
						fullDataSet.insert(feature.front());
					}
				}
				vector<string> fDS(fullDataSet.size());
				copy(fullDataSet.begin(), fullDataSet.end(), fDS.begin());
				int testInstLength  = (fullDataSet.size() / 10)  + 1;
				int count = 0;
				vector<string> testSet;
				set<string> testObj;
				while (count < testInstLength) {
					srand (time(NULL));
					int  iSecret = rand() % fullDataSet.size();
					string data = fDS.at(iSecret);
					vector <string> feature = split_string(data,"/");
					if (testObj.find(feature.front()) == testObj.end()) {
						testObj.insert(feature.front());
						testSet.push_back(data);
						count++;
					}
				}
				for(unsigned i = 0;i < testSet.size(); i++) {
					//       cout << testSet.at(i) << " , ";
				}
				//   exit(0);
				return testSet;
			}

                        vector<string> prepareTestDataJack(vector<string> trainCls) {
                                map<string, set<string>> fullDataSet;
                                for(unsigned i = 0;i < trainCls.size(); i++) {
                                        if(trainCls.at(i) != "") {
                                                string trainfileData = trainCls.at(i);
                                                vector <string> feature = split_string(trainfileData,",");
						vector <string> feature1 = split_string(feature.front(),"/");
						string cat = feature1.front();
						string inst = feature.front();
						if (fullDataSet.find(cat) == fullDataSet.end() ) {
							set<string> insts;
							insts.insert(inst);
                                                        fullDataSet[cat] = insts;
						} else {
							set<string> insts = fullDataSet[cat];
							insts.insert(inst);
							fullDataSet[cat] = insts;
						}
                                        }
                                }
				vector<string> testSet;
				for (map<string,set<string>>::iterator it=fullDataSet.begin(); it!=fullDataSet.end(); ++it) {
                                   set<string> itA = it->second;
				   string itB = it->first;
				   vector<string> fDS(itA.size());
				   copy(itA.begin(), itA.end(), fDS.begin());
				   int testInstLength  =  1;
				   int count = 0;
				   while (count < testInstLength) {
                                        srand (time(NULL));
                                        int  iSecret = rand() % itA.size();
                                        string data = fDS.at(iSecret);
					testSet.push_back(data);
					count++;
				   }
				}
                                for(unsigned i = 0;i < testSet.size(); i++) {
                                        //       cout << testSet.at(i) << " , ";
				}
                                return testSet;
                        }

			vector<string> prepareTestDataAL(vector<string> trainCls, int interval,int intervalLimit) {


				vector<string> testSet;
				for(unsigned i = 0;i < trainCls.size(); i++) {
					if(trainCls.at(i) != "") {
						if (i > interval and i < intervalLimit) {
							string trainfileData = trainCls.at(i);
							vector <string> feature = split_string(trainfileData,",");
							string data =  feature.front();
							testSet.push_back(data);
						}
					}
				}
				std::set<std::string> s(testSet.begin(), testSet.end());
				vector<string> testSet1;
				std::copy(s.begin(), s.end(), std::back_inserter(testSet1));
				return testSet1;
			}

			vector<string> prepareTrainData(vector<string> trainCls,vector<string> testSet) {
				vector<string> trainDS;
				for(unsigned i = 0;i < trainCls.size(); i++) {
					if(trainCls.at(i) != "") {
						string trainfileData = trainCls.at(i);
						vector <string> feature = split_string(trainfileData,",");
						if (find(testSet.begin(), testSet.end(), feature.front()) == testSet.end()) {
							//             if (testSet.find(feature.front()) == testSet.end()) {
							trainDS.push_back(trainfileData);
						}
						}
					}

					return trainDS;
				}

				vector<string> prepareTrainDataAL(vector<string> trainCls,int interval) {
					vector<string> trainDS;
					for(unsigned i = 0;i < interval; i++) {
						if(trainCls.at(i) != "") {
							string trainfileData = trainCls.at(i);

							trainDS.push_back(trainfileData);

						}
					}

					return trainDS;
				}

				map<string,vector<string>> prepareNegSetsFromInstances(string dsLoc,vector<string> trainCls,vector<string> testSet) {
					map<string,vector<string>> negMapInstances;
					for(unsigned i = 0;i < trainCls.size(); i++) {
						if(trainCls.at(i) != "") {
							string trainfileData = trainCls.at(i);
							vector <string> feature = split_string(trainfileData,":");
							string ds = feature.front();
							ds.erase (std::remove (ds.begin(), ds.end(), ' '), ds.end());
							vector <string> labels = split_string(feature.back(),",");
							vector <string> filteredLabels;
							for(unsigned j = 0;j < labels.size(); j++) {
								string lbl = labels.at(j);
								lbl.erase (std::remove (lbl.begin(), lbl.end(), ' '), lbl.end());
								if (find(testSet.begin(), testSet.end(), lbl) == testSet.end()) {
									//filteredLabels.push_back(dsLoc + "/" + lbl);
									filteredLabels.push_back(lbl);
								}
							}
							vector<string>::iterator it;
							it = find (filteredLabels.begin(), filteredLabels.end(), "");
							if (it != labels.end()) {
								filteredLabels.erase(it);
							}
							//  cout << ds << " ---> ";
							negMapInstances[ds] = filteredLabels;
							for(unsigned j = 0;j < filteredLabels.size(); j++) {
								string lbl = filteredLabels.at(j);
							//        cout << lbl << ",";
							}
					//		    cout << endl;
						}
					}
					for(unsigned j = 0;j < testSet.size(); j++) {
						string lbl = testSet.at(j);
						//    cout << lbl << ",";
					}
					return negMapInstances;
				}

				map<string,vector<string>> prepareNegSets(string dsLoc,vector<string> trainCls) {
					map<string,vector<string>> negMapInstances;
					map<string,vector<string>> fullDataSet;
					for(unsigned i = 0;i < trainCls.size(); i++) {
						if(trainCls.at(i) != "") {
							string trainfileData = trainCls.at(i);
							vector <string> feature = split_string(trainfileData,",");
							string ds = feature.front();
							vector <string> labels = split_string(feature.back()," ");
							vector<string>::iterator it;
							it = find (labels.begin(), labels.end(), "");
							if (it != labels.end()) {
								labels.erase(it);
							}
							if (fullDataSet.find(ds) == fullDataSet.end() ) {
								fullDataSet[ds] = labels;
							} else {
								vector<string> s = fullDataSet[ds];
								s.insert(s.end(),labels.begin(), labels.end());
								fullDataSet[ds] = s;
							}            
						}
					}
					// show content:
					for (map<string,vector<string>>::iterator it=fullDataSet.begin(); it!=fullDataSet.end(); ++it) {
						vector<string> itA = it->second;
						sort(itA.begin(), itA.end());
						//          cout << it->first << " => ";
						negMapInstances[it->first];
						for (map<string,vector<string>>::iterator itSec=fullDataSet.begin(); itSec!=fullDataSet.end(); ++itSec) {
							if(it->first != itSec->first ) {
								vector<string> itB = itSec->second;
								sort(itB.begin(), itB.end());
								vector<string> v;
								set_intersection(itA.begin(),itA.end(),itB.begin(),itB.end(), back_inserter(v));
								if(v.size() == 0) {
									negMapInstances[it->first].push_back(dsLoc + "/" + itSec->first);
									//                 cout << itSec->first << "(";
									//                cout  << itSec->first << ", " ;
									for(unsigned i = 0;i < v.size(); i++) {  
										//                  cout << v.at(i) << ",";
									}
									//                cout << ") ";
								}       
							}
						}
						//	  cout << endl;
					}
					return negMapInstances;
				}


				void negLabelUpdateTFIDF(map<string,vector<string>> negMapInstances,vector<string> anntn,string inst) {

					vector<string> negs = negMapInstances[inst];
					for(vector<string>::iterator itan = anntn.begin(); itan != anntn.end(); ++itan) {
						string lbl ;
						lbl = *itan;
						if(labelIndMat.find(lbl) == labelIndMat.end()) {
							labelIndMat[lbl] = negs;
						} else {
							vector<string> s = labelIndMat[lbl] ;
							s.insert(s.end(),negs.begin(), negs.end());
							labelIndMat[lbl] = s;
						}
					}    
					for(vector<string>::iterator itan = negs.begin(); itan != negs.end(); ++itan) {
						string lbl ;
						lbl = *itan;
						//    cout << lbl << " ";
					}
				}

				void folderByfolderExecution(string dsLoc,string trainLocation,string trainConfFile,string testConfFile,string libLinearLoc,bool syn,int negType) {
					testAnnotation = {};
					prepAnnotation(testAnnot,fullAnnot);
					vector<string> trainCls = fileToVector(trainConfFile);
					vector<string> trainClsCopy = trainCls;
					//cout << trainCls.size() << endl;
					vector<string> testSet = prepareTestData(trainCls);
					trainCls = prepareTrainData(trainCls,testSet); 
					// cout << trainCls.size() << endl;
					map<string,vector<string>> negMapInstances;
					if(negType == 2) {
						//       string negFile  = "/home/npillai1/AL/ConfFile/3k_tfidf_filtered_negativedataset.conf";
						//       string negFile  = "/home/npillai1/AL/ConfFile/3k_tfidf_N10_filtered_fulldataset.conf";
						string negFile = trainConfFile;
						vector<string> tnC = fileToVector(negFile);
						tnC = prepareTrainData(tnC,testSet);
						negMapInstances = prepareNegSets(dsLoc,tnC);

					} else if (negType == 3) {
						string negFile  = "/home/npillai1/AL/ConfFile/negLabels.log";
						vector<string> tnC = fileToVector(negFile);
						negMapInstances = prepareNegSetsFromInstances(dsLoc,tnC,testSet);
						negType = 2;
					}
					int index = 0;
					while(trainCls.size() > 0 ) {
						string trainfileData = trainCls.at(index);
						trainCls.erase(trainCls.begin() + index);
						// cout << trainfileData << endl;
						vector <string> feature = split_string(trainfileData,",");
						string objImgLoc = dsLoc + "/" + feature.front();
						string annotation = feature.back();
						//      cout << endl << annotation << ", ";
						//   vector<string> anntn1 = processLanguage(annotation);
						vector<string> anntn1 = split_string(annotation," ");
						vector<string> anntn;
						for(unsigned i = 0;i < anntn1.size(); i++) {
							if(anntn1.at(i) != "") {
								anntn.push_back(anntn1.at(i));
							}
						}
						vector<string> an1 = anntn;
						// cout << objImgLoc << endl;
						for(vector<string>::iterator itan = anntn.begin(); itan != anntn.end(); ++itan) {
							string lbl ;
							lbl = *itan;
							posLabelMapUpdate(lbl,objImgLoc);
							if(negType == 1) {
								negLabelMapUpdate(an1,lbl);
							}
						}
						if (negType == 2) {
							negLabelUpdateTFIDF(negMapInstances,anntn,feature.front());     
						}
					}
					/*
					   for(map<string,vector<string>>::iterator itit=posLabelIndMat.begin();itit != posLabelIndMat.end();++itit) {
					   ostringstream dat;
					   dat << itit->first;
					   string lbl = dat.str();
					   vector<string> poss = posLabelIndMat[lbl];
					   vector<string> negs = labelIndMat[lbl];
					   cout << lbl << " -----> " << endl;
					   cout << "Pos -> ";
					   for(vector<string>::iterator itan = poss.begin(); itan != poss.end(); ++itan) {
					   cout << *itan << " ";
					   }
					   cout << endl << "Neg -> ";
					   for(vector<string>::iterator itan = negs.begin(); itan != negs.end(); ++itan) {
					   cout << *itan << " ";
					   }
					   cout << endl;

					   }
					   */
					//// testing /////
					string tempTestLoc = dsLoc + "/testtest";
					//    vector<int> fileIndexes = prepTestFolder(dsLoc,testConfFile,tempTestLoc);
					vector<int> fileIndexes = prepareTestFolders(dsLoc,testSet,tempTestLoc);
					vector<string> testLabels = saveTestTrueValues(testSet,tempTestLoc,trainClsCopy);
					addFoldersAndTestAll(trainLocation,tempTestLoc,libLinearLoc,testLabels,syn,negType);
				}

				void   fullmanual10CrossExecutionAndPrediction(string dsLoc,string trainLocation,string trainConfFile,string testConfFile,string libLinearLoc,bool syn) {
					int interval = 25;
					testAnnotation = {};
					prepAnnotation(testAnnot,fullAnnot);

					vector<string> trainCls = fileToVector(trainConfFile);
					//    set<string> fhits;
					//    for(unsigned i = 0; i < trainCls.size();i++) {
					//        string s = trainCls.at(i);
					//        vector <string> feature = split_string(s,","); 
					//        fhits.insert(feature.front());
					//    }
					//    vector<string> trainFlds;
					//    copy(fhits.begin(), fhits.end(), inserter(trainFlds,trainFlds.begin()));
					//   vector<string> testFlds;
					//    int testNo = fhits.size() / 10;
					//    for(unsigned i = 0; i < testNo; i++) {
					//     srand(time(NULL));
					//      int index = rand() % trainFlds.size();
					//      testFlds.push_back(trainFlds.at(index));
					//      trainFlds.erase(trainFlds.begin() + index);
					//      sleep(2);
					//    } 
					//    string confNew = trainConfFile + "-new";
					//    string cmd = ">"+confNew;
					//    int ret = system(cmd.c_str());
					//   cmd = ">"+testConfFile;
					//  ret = system(cmd.c_str());
					//   for(unsigned i = 0; i < trainFlds.size();i++) {   
					//       cmd = "cat " + trainConfFile + " |grep '" + trainFlds.at(i) + ",' >> " + confNew;
					//      ret = system(cmd.c_str()); 
					//   }
					//   for(unsigned i = 0; i < testFlds.size();i++) {
					//       cmd = "cat " + trainConfFile + " |grep '" + testFlds.at(i) + ",' |tail -n 1 >> " + testConfFile;
					//       ret = system(cmd.c_str());
					//   }
					//    trainCls = fileToVector(confNew);
					interval = trainCls.size();
					int count = 0;
					//    while(trainCls.size() > 0 ) {
					while(count < 3) {
						//       for(unsigned i = 0; i < interval;i++) {
						//          if(trainCls.size() > 0) {
						//	     srand(time(NULL));
						//          int index = rand() % trainCls.size();
						int index = 0;
						count = count + 1;
						trainCls = randomTrainAddition(dsLoc,trainCls,index,trainLocation,syn);
						//          }
						cout << count << endl;
					}
					cout << "Train Instance Number :: " << count  << endl;
					testWithTestFoldersAndPredict(dsLoc,testConfFile,trainLocation,libLinearLoc,syn,count,"manual");
					//    }

				}
				void alBatchModel(string dsLoc,string trainLocation,string trainConfFile,string testConfFile,string libLinearLoc,bool syn) {
					int negType = 3;
					int negTypeDefault = 3;
					cout << "Batch mode started" << endl;
					testAnnotation = {};
					prepAnnotation(testAnnot,fullAnnot);
					vector<string> trainCls = fileToVector(trainConfFile);
					vector<string> trainClsCopy = trainCls;
					//cout << trainCls.size() << endl;
					vector<string> testSet = prepareTestData(trainCls);
					trainCls = prepareTrainData(trainCls,testSet); 
					trainClsCopy = trainCls;
					// cout << trainCls.size() << endl;
					map<string,vector<string>> negMapInstances;
					if(negType == 2) {
						//       string negFile  = "/home/npillai1/AL/ConfFile/3k_tfidf_filtered_negativedataset.conf";
						//       string negFile  = "/home/npillai1/AL/ConfFile/3k_tfidf_N10_filtered_fulldataset.conf";
						string negFile = trainConfFile;
						vector<string> tnC = fileToVector(negFile);
						tnC = prepareTrainData(tnC,testSet);
						negMapInstances = prepareNegSets(dsLoc,tnC);

					} else if (negType == 3) {
						string negFile  = "/home/npillai1/AL/ConfFile/negLabels.log";
						vector<string> tnC = fileToVector(negFile);
						negMapInstances = prepareNegSetsFromInstances(dsLoc,tnC,testSet);
						negType = 2;
					}
					//int interval = 1000;
					int interval  = 500;
					bool repFlag = false;
					int startInterval = 0;
					while(repFlag == false) {
						trainCls = trainClsCopy;
						if (trainCls.size() <= startInterval + interval) {
							startInterval = trainCls.size();
							repFlag = true;
						} else {
							startInterval = startInterval + interval;
						}

						int index = 0;
						ostringstream dat;
						dat << trainLocation << "/"  << startInterval << "/AL";
						string trainLocation1 = dat.str();
						cout << "starting for interval " << startInterval << endl;
						int interval1 = startInterval;

						int intervalAL = startInterval / 2;
						vector<string> testSetAL = prepareTestDataAL(trainCls,intervalAL, startInterval);
						vector<string> trainClsAL = prepareTrainDataAL(trainCls,intervalAL); 
						vector<string> trainClsCopyAL = trainClsAL;
						cout << "Train test sets " << endl;
						cout << trainClsCopyAL.size() << endl;
						cout << testSetAL.size() << endl;
						map<string,vector<string>> negMapInstancesAL;
						int intervalAL1 = intervalAL;	
						labelIndMat.clear();
						posLabelIndMat.clear();
						index = 0;
						while(intervalAL1 > 0 ) {
							intervalAL1 = intervalAL1 - 1;  
							string trainfileDataAL = trainClsAL.at(index);
							trainClsAL.erase(trainClsAL.begin() + index);
							// cout << trainfileData << endl;
							vector <string> feature = split_string(trainfileDataAL,",");
							string objImgLoc = dsLoc + "/" + feature.front();
							string annotation = feature.back();
							//      cout << endl << annotation << ", ";
							//   vector<string> anntn1 = processLanguage(annotation);
							vector<string> anntn1 = split_string(annotation," ");
							vector<string> anntn;
							for(unsigned i = 0;i < anntn1.size(); i++) {
								if(anntn1.at(i) != "") {
									anntn.push_back(anntn1.at(i));
								}
							}
							vector<string> an1 = anntn;
							// cout << objImgLoc << endl;
							for(vector<string>::iterator itan = anntn.begin(); itan != anntn.end(); ++itan) {
								string lbl ;
								lbl = *itan;
								posLabelMapUpdate(lbl,objImgLoc);
								if(negType == 1) {
									negLabelMapUpdate(an1,lbl);
								}
							}
							if (negType == 2) {
								negLabelUpdateTFIDF(negMapInstancesAL,anntn,feature.front());     
							}
						}
						//// testing /////
						string tempTestLoc = dsLoc + "/testtest";
						//    vector<int> fileIndexes = prepTestFolder(dsLoc,testConfFile,tempTestLoc);
						vector<int> fileIndexesAL = prepareTestFolders(dsLoc,testSetAL,tempTestLoc);
						vector<string> testLabelsAL = saveTestTrueValues(testSetAL,tempTestLoc,trainClsCopyAL);
						int correctNo = addFoldersAndTestAllAL(trainLocation1,tempTestLoc,libLinearLoc,testLabelsAL,syn,negType);

						/// clearing the learning and adding contents to real testing	


						trainCls = trainClsCopy;
						interval1 = startInterval + (correctNo / 4);

						ostringstream dat1;
						dat1 << trainLocation << "/"  << startInterval << "/Test";
						trainLocation1 = dat1.str();
						if(negTypeDefault == 2) {
							//       string negFile  = "/home/npillai1/AL/ConfFile/3k_tfidf_filtered_negativedataset.conf";
							//       string negFile  = "/home/npillai1/AL/ConfFile/3k_tfidf_N10_filtered_fulldataset.conf";
							string negFile = trainConfFile;
							vector<string> tnC = fileToVector(negFile);
							tnC = prepareTrainData(tnC,testSet);
							negMapInstances = prepareNegSets(dsLoc,tnC);
							negType = 2;
						} else if (negTypeDefault == 3) {
							string negFile  = "/home/npillai1/AL/ConfFile/negLabels.log";
							vector<string> tnC = fileToVector(negFile);
							negMapInstances = prepareNegSetsFromInstances(dsLoc,tnC,testSet);
							negType = 2;
						}



						labelIndMat.clear();
						posLabelIndMat.clear();
						index = 0;

						/// AL Testing////	
						cout << "new interval " << interval1 << endl;		
						while(interval1 > 0 ) {
							interval1 = interval1 - 1;  
							string trainfileData = trainCls.at(index);
							trainCls.erase(trainCls.begin() + index);
							// cout << trainfileData << endl;
							vector <string> feature = split_string(trainfileData,",");
							string objImgLoc = dsLoc + "/" + feature.front();
							string annotation = feature.back();
							//      cout << endl << annotation << ", ";
							//   vector<string> anntn1 = processLanguage(annotation);
							vector<string> anntn1 = split_string(annotation," ");
							vector<string> anntn;
							for(unsigned i = 0;i < anntn1.size(); i++) {
								if(anntn1.at(i) != "") {
									anntn.push_back(anntn1.at(i));
								}
							}
							vector<string> an1 = anntn;
							// cout << objImgLoc << endl;
							for(vector<string>::iterator itan = anntn.begin(); itan != anntn.end(); ++itan) {
								string lbl ;
								lbl = *itan;
								posLabelMapUpdate(lbl,objImgLoc);
								if(negType == 1) {
									negLabelMapUpdate(an1,lbl);
								}
							}
							if (negType == 2) {
								negLabelUpdateTFIDF(negMapInstances,anntn,feature.front());     
							}
						}
						//// testing /////
						tempTestLoc = dsLoc + "/testtest";
						//    vector<int> fileIndexes = prepTestFolder(dsLoc,testConfFile,tempTestLoc);
						vector<int> fileIndexes = prepareTestFolders(dsLoc,testSet,tempTestLoc);
						vector<string> testLabels = saveTestTrueValues(testSet,tempTestLoc,trainClsCopy);

						cout << endl << "----------------------------------------------------------------------------" << endl;
						cout << startInterval << "  Nisha ----> Real testing " << endl;	
						addFoldersAndTestAll(trainLocation1,tempTestLoc,libLinearLoc,testLabels,syn,negType);
					}
				}

				vector<string> uncertainInstancesForActiveLabeling(string csvFile,string featureType,int execMode, int extraNegPoints) {
                                        fstream fs(csvFile,fstream::in);
                                        string line;
                                        getline(fs,line,'\0');
                                        vector<string> labels ;
                                        string bck_line = line;
                                        vector <string> testCls = split_string(line,"\n");
					vector<string> uncertainPoints;
                                        for(unsigned inst = 1;inst < testCls.size(); inst++) {
                                                string trainfileData = testCls.at(inst);
                                                if(trainfileData != "") {
                                                        vector<string> values = split_string(trainfileData,",");
                                                        labels.push_back(values.at(0));
                                                        //       cout << values.at(0) << endl;
						}
					}
					string testNos = testCls.at(0);
					vector<double> probMatches;
                                        vector<string> tests = split_string(testNos,",");
                                        for(unsigned inst1 = 1;inst1 < tests.size(); inst1++) {
                                                string test_n = tests.at(inst1);
                                                vector<double> testProbs;
                                                if(test_n != "") {
                                                        unsigned index = 0;
                                                        string::iterator end_pos = remove(test_n.begin(), test_n.end(), ' ');
                                                        test_n.erase(end_pos, test_n.end());
                                                        testCls = split_string(bck_line,"\n");
                                                        for(unsigned inst = 1;inst < testCls.size(); inst++) {
                                                                string trainfileData = testCls.at(inst);
                                                                if(trainfileData != "") {
                                                                        vector<string> values = split_string(trainfileData,",");
                                                                        double prob = 0.0;
                                                                        if(values.size() > 2) {
                                                                                string val = values.at(inst1);
                                                                                prob = atof(val.c_str());
                                                                        }
                                                                        testProbs.push_back(prob);
                                                                }
                                                        }
							vector<double> entrIndices = findEntropyTotal(testProbs);
							bool uncertain = false;
							for (int i = 0; i < entrIndices.size(); i++) {
								cout << "ENTR " << testProbs.at(i) << " - " << entrIndices.at(i) << endl;
									if(entrIndices.at(i) > 0.95) {
										uncertain = true;
									}	
									cout << uncertain << endl;
							}
							if(uncertain == true) {
								uncertainPoints.push_back(test_n);
							}
						}
					}
					cout << "SIZE " << uncertainPoints.size() << endl;
					return uncertainPoints;
				}

				vector<string> verifyFullPredictionOneFeature(string csvFile,string featureType,int execMode, int extraNegPoints) {
				         vector<string> uncertainPts ;
					 unctFMeasurePoints.clear();
					//cout << testAnnotation.size() << endl;
					fstream fs(csvFile,fstream::in);
					string line;
					getline(fs,line,'\0');
					vector<string> labels ;
					string bck_line = line;
					vector <string> testCls = split_string(line,"\n");
                                   
					for(unsigned inst = 1;inst < testCls.size(); inst++) {
						string trainfileData = testCls.at(inst);
						if(trainfileData != "") {
							vector<string> values = split_string(trainfileData,",");
							labels.push_back(values.at(0));
							//       cout << values.at(0) << endl;
						}
					}
					string testNos = testCls.at(0);
					vector<double> probMatches;
					if (extraNegPoints == 1) {
							additionalNegInstances.clear();
					}
					vector<string> tests = split_string(testNos,",");
					for(unsigned inst1 = 1;inst1 < tests.size(); inst1++) {
						string test_n = tests.at(inst1);
						vector<double> testProbs;
						if(test_n != "") {
							unsigned index = 0;
							double probs = 0.0;
							string::iterator end_pos = remove(test_n.begin(), test_n.end(), ' ');
							test_n.erase(end_pos, test_n.end());
							string sample = "";
							double entrThresh = 0.0;
							if (featureType == "rgb") {
								sample = colorAnnotation[test_n];
                                                                probs = 0.70;
								entrThresh = 1.25;
							} else if (featureType == "shape") {
								sample = shapeAnnotation[test_n];
                                                                probs = 0.50;
								entrThresh = 2.0;
							} else if (featureType == "object") {
								sample = objectAnnotation[test_n];
                                                                probs = 0.50;
								entrThresh = 2.0;
							} 

							testCls = split_string(bck_line,"\n");
							vector<int> maxIndexes1;
							for(unsigned inst = 1;inst < testCls.size(); inst++) {
								string trainfileData = testCls.at(inst);
								if(trainfileData != "") {
									vector<string> values = split_string(trainfileData,",");
									double prob = 0.0;
									if(values.size() > 2) {
										string val = values.at(inst1);
										prob = atof(val.c_str());
										if(prob > probs) {
											maxIndexes1.push_back(inst - 1);
										}
										
									}
									testProbs.push_back(prob);
								}
							}
	
							vector<int> maxIndexes;
							string mchRes;
							 if (execMode == 1) {
								mchRes = "echo '" + csvFile + " --> "  + test_n + "( " + sample + "  ) predicted as " ;
								for(unsigned inst2 = 0;inst2 < maxIndexes1.size(); inst2++) {
									mchRes += " " + labels.at(maxIndexes1.at(inst2));
								}
								maxIndexes = maxIndexes1;
								cout << "PREDICTION -> " << mchRes << endl;
							} else if (execMode == 2) {
								vector<int> entrIndices = findEntropyAL(testProbs,entrThresh);    
								mchRes = "echo '" + csvFile + " --> "  + test_n + "( " + sample + "  ) predicted as " ;
								for(unsigned inst2 = 0;inst2 < entrIndices.size(); inst2++) {
									mchRes += " " + labels.at(entrIndices.at(inst2));	
								}
								maxIndexes = entrIndices;
								cout << "ENTROPY PREDICTION -> " << mchRes << endl;
							}
							//      int ret = system(mchRes.c_str());
							//cout << sample << endl;
							vector<string> sampleLabels = split_string(sample," ");
							
							vector<string> truePos;						
							vector<string> falsePos;
							vector<string>  falseNeg;
							bool found = false;
							vector<string> neginstWords;
							for(unsigned inst2 = 0;inst2 < maxIndexes.size(); inst2++) {
								string lbl = labels.at(maxIndexes.at(inst2));
								end_pos = remove(lbl.begin(), lbl.end(), ' ');
								lbl.erase(end_pos, lbl.end());
								bool checkNegWord = false;
								for(unsigned inst1 = 0;inst1 < sampleLabels.size(); inst1++) {
									string ss = sampleLabels.at(inst1);
									end_pos = remove(ss.begin(), ss.end(), ' ');
									ss.erase(end_pos, ss.end());
									if (sampleLabels.at(inst1) == lbl) {
										found = true;
										checkNegWord = true;
										truePos.push_back(lbl);
									}
								}
								//add to another map 
								if(checkNegWord == false) {
									neginstWords.push_back(lbl);
									falsePos.push_back(lbl);
								}
								
							}


							for(unsigned inst1 = 0;inst1 < sampleLabels.size(); inst1++) {
                                                        	string ss = sampleLabels.at(inst1);
								end_pos = remove(ss.begin(), ss.end(), ' ');
								ss.erase(end_pos, ss.end());
								bool check = false;
								for(unsigned inst2 = 0;inst2 < maxIndexes.size(); inst2++) {
        	                                                        string lbl = labels.at(maxIndexes.at(inst2));
	                                                                end_pos = remove(lbl.begin(), lbl.end(), ' ');
                                                                	lbl.erase(end_pos, lbl.end());
									if(ss == lbl) {
										check = true;
									}
								}
								if(check == false) {
									falseNeg.push_back(ss);
								}
							}
							cout << "TP " << truePos.size() << " FP " << falsePos.size() << " FN " << falseNeg.size() << endl;
							float prec = float(truePos.size()) /float(truePos.size() + falsePos.size());
							float recall = float(truePos.size()) / float(truePos.size() + falseNeg.size());
							cout << " PRECISION " << prec << ", RECALL " << recall << endl;
							/*
							if(prec != 0.0 and recall != 0.0) {
								float fMeasure = 2.0 * prec * recall / (prec + recall);
								cout << "F-Measure for " << test_n << " - " << fMeasure << endl;
								if(fMeasure < 0.5) {
									unctFMeasurePoints.push_back(test_n);
								}	
							} else {
								unctFMeasurePoints.push_back(test_n);
							}
							
							if(prec < 0.25) {
								unctFMeasurePoints.push_back(test_n);
							}
							*/
							if (extraNegPoints == 1) {
							   if(neginstWords.size() > 0) { 
  								map<string,vector<string>>::iterator it = additionalNegInstances.find(test_n);
 								if (it != additionalNegInstances.end()) {
									vector<string> tempAr = additionalNegInstances[test_n];
									tempAr.insert(tempAr.end(), neginstWords.begin(), neginstWords.end());
    									additionalNegInstances[test_n] = tempAr;
								} else {
									additionalNegInstances[test_n]  = neginstWords;
								}
							    }
							}
							// for each word in labels, check if that appears in whole list with that instance, else add that instance as negative sample for the word. 

							if(found){
								cout << "PREDICTED-> CORRECT " << endl;
							} else  {
								cout << "PREDICTED-> WRONG " << endl;  
                						uncertainPts.push_back(test_n);      
							}
						}
					}
					unctFMeasurePoints = uncertainPts;
					return uncertainPts;
				}

				string  addFoldersAndTestAllALOneFeature(string trainLocation,string tempTestLoc,string libLinearLoc,vector<string> testLabels,bool syn,int negType,string featureType,int execMode,int extraNegPoints) {
					string cmd = "mkdir -p " + trainLocation;
					int ret = system(cmd.c_str());
					string csvFile = "";
						if (featureType == "rgb") {
							csvFile = trainLocation + "/rgbConfusionMatrix.csv";
						} else if  (featureType == "shape") {
							csvFile = trainLocation + "/shapeConfusionMatrix.csv";
						} else if (featureType == "object") {
							csvFile = trainLocation + "/objectConfusionMatrix.csv";
						} 
					cmd = ">"+ csvFile;
					ret = system(cmd.c_str());
					cout << "-------CSV FILE--------------- " <<  csvFile << endl;

					string lbls = ",";
					for (int i = 0; i < testLabels.size() ; i ++ ) {
						lbls += testLabels.at(i) + ",";
					}
					cout << "--->NASH-->" << lbls << endl;
					cmd = "echo " + lbls +">>" + csvFile;
					ret = system(cmd.c_str());

					cout << "All Words -> ";
					for(map<string,vector<string>>::iterator itit=posLabelIndMat.begin();itit != posLabelIndMat.end();++itit) {
						ostringstream dat;
						dat << itit->first;
						string lbl = dat.str();
						cout << lbl << ", ";
					}
					cout << endl << endl;
					ofstream myfile;
					myfile.open(negPosInstances);
					myfile << endl;
					myfile.close();
					for(map<string,vector<string>>::iterator itit=posLabelIndMat.begin();itit != posLabelIndMat.end();++itit) {
						ostringstream dat;
						dat << itit->first;
						string lbl = dat.str();
						cout << lbl << " is starting execution !!!!!" << endl;
						//         if (lbl == "object" or lbl == "block" or lbl == "box" or lbl == "brick" or lbl == "fruit" or lbl == "image" or lbl == "item" or lbl == "photo" or lbl == "picture" or lbl == "thing" or lbl == "toy" or lbl == "vegetable" or lbl == "woord" or lbl == "wooden" ) {
						cout << endl << " >>>>>>  " << lbl << "--->";
						vector<string> poss = posLabelIndMat[lbl];
						vector<string> negss = labelIndMat[lbl];
						set<string> snegs(negss.begin(), negss.end());
						vector<string> negs(snegs.begin(), snegs.end());
						cout <<  "Pos # " << poss.size() << ", Neg # " << negs.size() << endl;
						if (poss.size() > minInstances and negs.size() > minInstances) {
							updateFolder(lbl,poss,negs,trainLocation,tempTestLoc,negType);
							prepFileLearnAndTest(trainLocation,lbl,libLinearLoc,csvFile,featureType);

							// if (lbl != "carrot") {           
							string newLabelLoc = trainLocation + "/" + lbl ;      
							cmd = "rm -rf " + newLabelLoc;
							ret = system(cmd.c_str());
							string rmFiles = trainLocation + "/" + lbl + "/*.linear";
							cmd = "rm -rf " + rmFiles;
							ret = system(cmd.c_str());
							rmFiles = trainLocation + "/" + lbl + "/test";
							cmd = "rm -rf " + rmFiles;
							ret = system(cmd.c_str());
							rmFiles = trainLocation + "/" + lbl + "/" + lbl + "*";           
							cmd = "rm -rf " + rmFiles;
							ret = system(cmd.c_str()); 
							//  }  
						}
						cout << endl << lbl << " - Completed execution !!!!!" << endl;
						//       }

					}
                                        return csvFile;
				}
				vector<int> updatedIndices(vector<string> uncertainDataPoints, vector<int> trainIndexPtrs, vector<string> trainCls,  int start, int end) {
					std::set<std::string> s(uncertainDataPoints.begin(), uncertainDataPoints.end());
					vector<string> uDPts;
					std::copy(s.begin(), s.end(), std::back_inserter(uDPts));
					for(unsigned i = 0;i < trainCls.size(); i++) {
						if(trainCls.at(i) != "") {
							if (i >= start and i < end ) {
								string trainfileData = trainCls.at(i);
								vector <string> feature = split_string(trainfileData,",");
								string data =  feature.front();
								string::iterator end_pos = remove(data.begin(), data.end(), ' ');
								data.erase(end_pos, data.end()); 
                                                                feature = split_string(data,"/");
                                                                data = feature.back();
                                                             
								if (std::find(uDPts.begin(), uDPts.end(), data) != uDPts.end()) {
									trainIndexPtrs.push_back(i);	
                                                                }
		
							}
						}
					}

                                        return trainIndexPtrs;
				}


				vector<int> addTrainIndexes(vector<int> indices , int start, int end) {
					for(int i = start;i < end;i++) {
						indices.push_back(i);
					}
					return indices;
				}

				map<string,vector<string>> findNegDataPointsforTerms(string dsLoc, vector<int> trainIndexPtrs,vector<string> trainCls) {	
					// here we have  a map with key as instance and values as 'words that are probably negative instances'
					//  function that iterates through all points in trainIndexPtrs and make a map of all words that instance contains. check those words and find out the difference from previous to this. Those are negative words for those instance. 
					// create a map that holds 'words and then insances as negative poiints
					// return the map

					map<string,vector<string>> instDescriptions;	
					map<string,string> instFullLocations;	
					map<string,int> instFrequencies;			
					for (int i = 0; i < trainIndexPtrs.size(); i++) {
						int index =  trainIndexPtrs.at(i);
						string trainfileDataAL = trainCls.at(index);

						vector <string> feature = split_string(trainfileDataAL,",");
						//string objImgLoc = dsLoc + "/" + feature.front();
						string instPoint = feature.front();
						string::iterator end_pos = remove(instPoint.begin(), instPoint.end(), ' ');
						instPoint.erase(end_pos, instPoint.end()); 
						string objImgLoc = dsLoc + "/" + instPoint;
						string annotation = feature.back();
							
						vector<string> anntn1 = split_string(annotation," ");
						vector<string> anntn;
						for(unsigned i = 0;i < anntn1.size(); i++) {
							if(anntn1.at(i) != "") {
								string ab = anntn1.at(i);
								string::iterator end_pos1 = remove(ab.begin(), ab.end(), ' ');
								ab.erase(end_pos1, ab.end());
								anntn.push_back(ab);
							}
						}
						if ( instPoint != "" and anntn.size() > 0) {
							feature = split_string(instPoint,"/");
							string instPoint1 = feature.back();
							map<string,vector<string>>::iterator it = instDescriptions.find(instPoint1);
  							if (it != instDescriptions.end()) {
								vector<string> abc = instDescriptions[instPoint1];
								abc.insert(abc.end(),anntn.begin(),anntn.end());
								instDescriptions[instPoint1] = abc;
								instFrequencies[instPoint1] += 1;
							} else {
								instDescriptions[instPoint1] = anntn;
								instFullLocations[instPoint1] = objImgLoc;
								instFrequencies[instPoint1] = 1;
							}
						}
					}

					map<string,vector<string>> termNegInstances;
					for(map<string,vector<string>>::iterator itit=instDescriptions.begin();itit != instDescriptions.end();++itit) {
						ostringstream dat;
						dat << itit->first;
						string inst = dat.str();	
						vector<string> existWords = itit->second;
						map<string,vector<string>>::iterator it = additionalNegInstances.find(inst);
  						if (it != additionalNegInstances.end()) {
							vector<string> possNegWords = additionalNegInstances[inst];
							for (int ij=0;ij < possNegWords.size(); ij++) {
								string word = possNegWords.at(ij);
								if(find(existWords.begin(), existWords.end(), word)!=existWords.end()){
									vector<string> negs;
									//for(int ijk=0;ijk< instFrequencies[inst];ijk++) {
										negs.push_back(instFullLocations[inst]);
									//}
									map<string,vector<string>>::iterator itNeg = termNegInstances.find(word);
									if(itNeg != termNegInstances.end()) {
										vector<string> abc = termNegInstances[word];
										abc.insert(abc.end(),negs.begin(),negs.end());
										termNegInstances[word] = abc;
									} else {
										termNegInstances[word];
										termNegInstances[word] = negs;
									}

								}
							}
						
						}
					
					}
					return termNegInstances;


				}
		 
				void negLabelUpdateExtraNegs(map<string,vector<string>> termNegInstances,vector<string> anntn){

					
					for(vector<string>::iterator itan = anntn.begin(); itan != anntn.end(); ++itan) {
						string lbl ;
						lbl = *itan;
						if(termNegInstances.find(lbl) == termNegInstances.end()) {
							vector<string> ss = termNegInstances[lbl];
							if(labelIndMat.find(lbl) == labelIndMat.end()) {
								labelIndMat[lbl] = ss;
							} else {
								vector<string> s = labelIndMat[lbl] ;
								s.insert(s.end(), ss.begin(), ss.end());
								labelIndMat[lbl] = s;
							}
						}
					}    


				}

				void alBatchModelOneFeature(string dsLoc,string trainLocation,string trainConfFile,string testConfFile,string libLinearLoc,bool syn, string featureType, int execMode, int extraNegPoints ) {
					int negType = 3;
					int negTypeDefault = 3;
					cout << "Batch mode started" << endl;
					testAnnotation = {};
					prepAnnotation(testAnnot,fullAnnot);
					vector<string> trainCls = fileToVector(trainConfFile);
					vector<string> trainClsCopy = trainCls;
					//cout << trainCls.size() << endl;
					vector<string> testSet = prepareTestData(trainCls);
					trainCls = prepareTrainData(trainCls,testSet); 
					trainClsCopy = trainCls;
					// cout << trainCls.size() << endl;
					map<string,vector<string>> negMapInstances;
					map<string,vector<string>> termNegInstances;
					if(negType == 2) {
						//       string negFile  = "/home/npillai1/AL/ConfFile/3k_tfidf_filtered_negativedataset.conf";
						//       string negFile  = "/home/npillai1/AL/ConfFile/3k_tfidf_N10_filtered_fulldataset.conf";
						string negFile = trainConfFile;
						vector<string> tnC = fileToVector(negFile);
						tnC = prepareTrainData(tnC,testSet);
						negMapInstances = prepareNegSets(dsLoc,tnC);

					} else if (negType == 3) {
						string negFile  = "/home/npillai1/AL/ConfFile/negLabels.log";
						vector<string> tnC = fileToVector(negFile);
						negMapInstances = prepareNegSetsFromInstances(dsLoc,tnC,testSet);
						negType = 2;
					}
					//int interval = 1000;
					int interval  = 100;
					bool repFlag = false;
					int startInterval = 0;
					int startPtr = 0;
					int endPtr = 0;
					vector<int> trainIndexPtrs;
					if (trainCls.size() <= endPtr + interval) {
							endPtr = trainCls.size();
							repFlag = true;
					} else {
							endPtr = endPtr + interval;
					}
                                        trainIndexPtrs = addTrainIndexes(trainIndexPtrs, startPtr , endPtr);         
					while(repFlag == false) {
						/*
						if (endPtr > 100) {
							repFlag = true;
						}
						*/
						trainCls = trainClsCopy;
						startPtr = endPtr;
						if (trainCls.size() <= endPtr + interval) {
							endPtr = trainCls.size();
							repFlag = true;
						} else {
							endPtr = endPtr + interval;
						}

						int index = 0;
						ostringstream dat;
						dat << trainLocation << "/"  << endPtr << "/AL";
						string trainLocation1 = dat.str();
						cout << "starting for interval " << endPtr << endl;
						int interval1 = endPtr - startPtr;

						vector<string> testSetAL = prepareTestDataAL(trainCls, startPtr, endPtr);
						

						cout << "train test sets " << endl;
						cout << trainIndexPtrs.size() << endl;
						cout << testSetAL.size() << endl;

						labelIndMat.clear();
						posLabelIndMat.clear();

						for (int i = 0; i < trainIndexPtrs.size(); i++) {
							index =  trainIndexPtrs.at(i);
							string trainfileDataAL = trainCls.at(index);

							// cout << trainfileData << endl;
							vector <string> feature = split_string(trainfileDataAL,",");
							string objImgLoc = dsLoc + "/" + feature.front();
							string annotation = feature.back();
							//      cout << endl << annotation << ", ";
							//   vector<string> anntn1 = processLanguage(annotation);
							vector<string> anntn1 = split_string(annotation," ");
							vector<string> anntn;
							for(unsigned i = 0;i < anntn1.size(); i++) {
								if(anntn1.at(i) != "") {
									anntn.push_back(anntn1.at(i));
								}
							}
							vector<string> an1 = anntn;
							// cout << objImgLoc << endl;
							for(vector<string>::iterator itan = anntn.begin(); itan != anntn.end(); ++itan) {
								string lbl ;
								lbl = *itan;
								posLabelMapUpdate(lbl,objImgLoc);
								if(negType == 1) {
									negLabelMapUpdate(an1,lbl);
								}
							}
							if (negType == 2) {
								negLabelUpdateTFIDF(negMapInstances,anntn,feature.front());     
								if(extraNegPoints == 1) {
                                                                        negLabelUpdateExtraNegs(termNegInstances,anntn);
								}			
							}
						}
						//// testing /////
						string tempTestLoc = dsLoc + "/testtest";
						//    vector<int> fileIndexes = prepTestFolder(dsLoc,testConfFile,tempTestLoc);
						vector<int> fileIndexesAL = prepareTestFolders(dsLoc,testSetAL,tempTestLoc);
						vector<string> testLabelsAL = saveTestTrueValues(testSetAL,tempTestLoc,trainCls);
						string csvFile = addFoldersAndTestAllALOneFeature(trainLocation1,tempTestLoc,libLinearLoc,testLabelsAL,syn,negType,featureType,execMode,extraNegPoints);
                                                vector<string> uncertainDataPoints  = uncertainInstancesForActiveLabeling(csvFile,featureType,execMode,extraNegPoints);

//                                                cout << "UNCERTAIN DP " <<   uncertainDataPoints.size() << endl;
						cout << "UNCERTAIN DP " << uncertainDataPoints.size() << endl;
/*
						 if(extraNegPoints == 1) {
							termNegInstances = findNegDataPointsforTerms(dsLoc, trainIndexPtrs, trainCls);
							
						}
*/
//                                                 int correctNo = testLabelsAL.size() - uncertainDataPoints.size();
                                                // cout <<  "Correct No of Prediction --> " << correctNo << " / " <<  testLabelsAL.size()  <<  " == " << float(correctNo /  testLabelsAL.size()) << endl;
                                                 cout << "Data POints Before " <<  trainIndexPtrs.size() << endl;
						 if(uncertainDataPoints.size() > 0) {
							trainIndexPtrs = updatedIndices(uncertainDataPoints, trainIndexPtrs, trainCls, startPtr , endPtr);
						}
                                                  cout << "DATA POINTS  :  Actual - " <<  endPtr << " ,  AL  Points - " <<  trainIndexPtrs.size();
						/// clearing the learning and adding contents to real testing	
                                            

						trainCls = trainClsCopy;
						ostringstream dat1;
						dat1 << trainLocation << "/"  << endPtr << "/TEST";
						trainLocation1 = dat1.str();

						labelIndMat.clear();
						posLabelIndMat.clear();

						for( int i = 0; i < trainIndexPtrs.size(); i++) {
							index =  trainIndexPtrs.at(i);
							string trainfileData = trainCls.at(index);
							vector <string> feature = split_string(trainfileData,",");
							string objImgLoc = dsLoc + "/" + feature.front();
							string annotation = feature.back();
							//      cout << endl << annotation << ", ";
							//   vector<string> anntn1 = processLanguage(annotation);
							vector<string> anntn1 = split_string(annotation," ");
							vector<string> anntn;
							for(unsigned i = 0;i < anntn1.size(); i++) {
								if(anntn1.at(i) != "") {
									anntn.push_back(anntn1.at(i));
								}
							}
							vector<string> an1 = anntn;
							// cout << objImgLoc << endl;
							for(vector<string>::iterator itan = anntn.begin(); itan != anntn.end(); ++itan) {
								string lbl ;
								lbl = *itan;
								posLabelMapUpdate(lbl,objImgLoc);
								if(negType == 1) {
									negLabelMapUpdate(an1,lbl);
								}
							}
							if (negType == 2) {
								negLabelUpdateTFIDF(negMapInstances,anntn,feature.front());     
								 if(extraNegPoints == 1) {
									negLabelUpdateExtraNegs(termNegInstances,anntn);
								 }
							}
						}
						//// testing /////
						tempTestLoc = dsLoc + "/testtest";

                                                // trainIndexPtrs = updatedIndices(uncertainDataPoints, trainIndexPtrs, trainCls, startPtr + intervalAL, endPtr);


						//    vector<int> fileIndexes = prepTestFolder(dsLoc,testConfFile,tempTestLoc);
						vector<int> fileIndexes = prepareTestFolders(dsLoc,testSet,tempTestLoc);
						vector<string> testLabels = saveTestTrueValues(testSet,tempTestLoc,trainClsCopy);

						cout << endl << "----------------------------------------------------------------------------" << endl;
						cout << endPtr << "  Nisha ----> Real testing " << endl;	
						string csvFileTest = addFoldersAndTestAllALOneFeature(trainLocation1,tempTestLoc,libLinearLoc,testLabels,syn,negType,featureType,execMode,0);
						vector<string> uncertainDataPoints1  = verifyFullPredictionOneFeature(csvFileTest,featureType,execMode,extraNegPoints);


                                                 int correctNo1 = testLabels.size() - uncertainDataPoints1.size();
                                                cout <<  "END --- Correct No of Prediction --> " << correctNo1 << " / " <<  testLabels.size()  <<  " == " << float(correctNo1) / float( testLabels.size()) << endl;

						//addFoldersAndTestAll(trainLocation1,tempTestLoc,libLinearLoc,testLabels,syn,negType);
					}
				}


				void mlBatchModel(string dsLoc,string trainLocation,string trainConfFile,string testConfFile,string libLinearLoc,bool syn) {
					int negType = 3;
					cout << "Batch mode started" << endl;
					testAnnotation = {};
					prepAnnotation(testAnnot,fullAnnot);
					vector<string> trainCls = fileToVector(trainConfFile);
					vector<string> trainClsCopy = trainCls;
					//cout << trainCls.size() << endl;
					vector<string> testSet = prepareTestDataJack(trainCls);
					trainCls = prepareTrainData(trainCls,testSet); 
					trainClsCopy = trainCls;
					// cout << trainCls.size() << endl;
					map<string,vector<string>> negMapInstances;
					if(negType == 2) {
						//       string negFile  = "/home/npillai1/AL/ConfFile/3k_tfidf_filtered_negativedataset.conf";
						//       string negFile  = "/home/npillai1/AL/ConfFile/3k_tfidf_N10_filtered_fulldataset.conf";
						string negFile = trainConfFile;
						vector<string> tnC = fileToVector(negFile);
						tnC = prepareTrainData(tnC,testSet);
						negMapInstances = prepareNegSets(dsLoc,tnC);

					} else if (negType == 3) {
						string negFile  = "/home/npillai1/AL/ConfFile/negLabels.log";
						vector<string> tnC = fileToVector(negFile);
						negMapInstances = prepareNegSetsFromInstances(dsLoc,tnC,testSet);
						negType = 2;
					}
					regularizedLogisticRegression(dsLoc,trainCls,testSet,negMapInstances,testFullAnnotation,trainLocation);
					exit(0);
					//int interval = 10;
					int interval  = 2000;
					bool repFlag = false;
					int startInterval = 0;
					while(repFlag == false) {
						negType = 4;
						trainCls = trainClsCopy;
						labelIndMat.clear();
						posLabelIndMat.clear();
						int index = 0;
						//	repFlag = true;
						if (trainCls.size() <= startInterval + interval) {
							startInterval = trainCls.size();
							repFlag = true;
						} else {
							startInterval = startInterval + interval;
						}
						ostringstream dat;
						dat << trainLocation << "/"  << startInterval;
						string trainLocation1 = dat.str();
						cout << "starting for interval " << startInterval << endl;
						int interval1 = startInterval;
						map<string,vector<string>> fullDataSetForNeg4;
						set<string> allTokens;
						while(interval1 > 0 ) {
							interval1 = interval1 - 1;  
							string trainfileData = trainCls.at(index);
							trainCls.erase(trainCls.begin() + index);
							// cout << trainfileData << endl;
							vector <string> feature = split_string(trainfileData,",");
							string objImgLoc = dsLoc + "/" + feature.front();
							string annotation = feature.back();
							//      cout << endl << annotation << ", ";
							//   vector<string> anntn1 = processLanguage(annotation);
							vector<string> anntn1 = split_string(annotation," ");
							vector<string> anntn;
							for(unsigned i = 0;i < anntn1.size(); i++) {
								if(anntn1.at(i) != "") {
									allTokens.insert(anntn1.at(i));
									anntn.push_back(anntn1.at(i));
								}
							}
							if(negType == 4) {
							   string ds = dsLoc + "/" + feature.front();
							   if (fullDataSetForNeg4.find(ds) == fullDataSetForNeg4.end() ) {
                                                                fullDataSetForNeg4[ds] = anntn;
							   } else {
                                                                vector<string> s = fullDataSetForNeg4[ds];
                                                                s.insert(s.end(),anntn.begin(), anntn.end());
                                                                fullDataSetForNeg4[ds] = s;
							   }
							}
							vector<string> an1 = anntn;
							// cout << objImgLoc << endl;
							for(vector<string>::iterator itan = anntn.begin(); itan != anntn.end(); ++itan) {
								string lbl ;
								lbl = *itan;
								posLabelMapUpdate(lbl,objImgLoc);
								if(negType == 1) {
									negLabelMapUpdate(an1,lbl);
								}
							}
							if (negType == 2) {
								negLabelUpdateTFIDF(negMapInstances,anntn,feature.front());     
							}
						}
						if(negType == 4) {
							labelIndMat.clear();
							for(set<string>::iterator itan = allTokens.begin(); itan != allTokens.end(); ++itan) {
								string label ;
								label = *itan;
								labelIndMat[label];
								for (map<string,vector<string>>::iterator it=fullDataSetForNeg4.begin(); it!=fullDataSetForNeg4.end(); ++it) {
									vector<string> itB = it->second;
									string itA = it->first;
									if (find(itB.begin(), itB.end(), label) == itB.end()) {
										labelIndMat[label].push_back(itA) ;
/*
										if(labelIndMat.find(label) == labelIndMat.end()) {
											labelIndMat[label] = [itA];
										} else {
											labelIndMat[label].push_back(itA) ;
										} */	
									}					
								}
							}
							negType = 2;
						}
	/*
						for (map<string,vector<string>>::iterator it=labelIndMat.begin(); it!=labelIndMat.end(); ++it) {
							vector<string> itB = it->second;
							string itA = it->first;
							cout << itA << endl;
							for(vector<string>::iterator itan = itB.begin(); itan != itB.end(); ++itan) {
                                                                string label ;
                                                                label = *itan;
								cout << label << ",";
							}
							cout << endl;
							
						}
						cout<< negType << endl;
*/						//// testing /////
						string tempTestLoc = dsLoc + "/testtest";
						//    vector<int> fileIndexes = prepTestFolder(dsLoc,testConfFile,tempTestLoc);
						vector<int> fileIndexes = prepareTestFolders(dsLoc,testSet,tempTestLoc);
						vector<string> testLabels = saveTestTrueValues(testSet,tempTestLoc,trainClsCopy);
						cout << startInterval << "  Nisha ---> real testing " << endl; 
						addFoldersAndTestAll(trainLocation1,tempTestLoc,libLinearLoc,testLabels,syn,negType);
						cout << "END:: Execution with interval " << startInterval << endl;
					}
				}

				void alEntropyExecuteModel(string dsLoc,string trainLocation,string trainConfFile,string testConfFile,string libLinearLoc,bool syn) {
					//    manualExecutionAndPrediction(dsLoc,trainLocation,trainConfFile,testConfFile,libLinearLoc,syn);
					//    //   fullmanual10CrossExecutionAndPrediction(dsLoc,trainLocation,trainConfFile,testConfFile,libLinearLoc,syn);
					//
					//    //     activeExecEntropyAndPrediction(dsLoc,trainLocation,trainConfFile,testConfFile,libLinearLoc,syn);
					//    //      int count  = prepareNegMap(trainConfFile);
					//    //    int count = 2000;
					//    //    testWithTestFoldersAndPredict(dsLoc,testConfFile,trainLocation,libLinearLoc,syn,count,"manual");
					//    //      int negType = 1; //traditional implemented method
					int negType = 3; // removal of test instances which has got common words            
					//                folderByfolderExecution(dsLoc,trainLocation,trainConfFile,testConfFile,libLinearLoc,syn,negType);
       				    mlBatchModel(dsLoc,trainLocation,trainConfFile,testConfFile,libLinearLoc,syn);
					//	alBatchModel(dsLoc,trainLocation,trainConfFile,testConfFile,libLinearLoc,syn);
					string type = "rgb";
                                         int execMode = 2;  // 1 - threshold, 2 - entropy
					  int extraNegPoints = 0; // 0 - Not needed , 1 - need to find extract negative data points
//					alBatchModelOneFeature(dsLoc,trainLocation,trainConfFile,testConfFile,libLinearLoc,syn,type,execMode,extraNegPoints);

				}
				//

				void convertBatchCSVToTestFile() {

					string tmp = "temp.csv";
					vector<string> sth = fileToVector("/home/npillai1/AL/Batch_2657888_batch_results.csv");
					for(unsigned i = 0; i < sth.size(); i++) {
						vector <string> feature = split_string(sth.at(i),",");
						string fle = "";
						for(unsigned ij = 0; ij < 4; ij++) {
							string sth1 = feature.at(ij);
							cout << sth1 << endl;
							vector<string> sr = split_string(sth1,"/");
							string st = sr.back();
							cout << sr.back() << " " << sr.back().length() << endl;
							sth1.erase(sth1.end()- st.length() - 1,sth1.end());
							cout << sth1 << endl;
							fle += sth1 + ",";
						}
						string cmd = "echo '" + fle + "' >> " + tmp;
						int ret = system(cmd.c_str());
					}

					exit(0);

				}
				//void justCheck() {
				int man ( int argc, char *argv[] ) {
					testAnnotation = {};
                                        prepAnnotation(testAnnot,fullAnnot);
					string file = "/home/npillai1/AL/images/ManualLabeling/500/rgbConfusionMatrix.csv-10000-0.05.log";
					cout << file << endl;
					int k = verifyFullPrediction(file,1);

				}
