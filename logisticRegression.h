#ifndef LOGISTICREGRESSION_H
#define LOGISTICREGRESSION_H
#include <fstream>
#include <vector>
#include <set>
#include <iostream>
#include <math.h>
#include <sstream>
#include <cmath>
#include <algorithm>
#include <ctime>        // std::time
#include <cstdlib>      // std::rand, std::srand
#include <map>
#include <boost/filesystem.hpp>
#include <boost/algorithm/string/predicate.hpp>
#include <boost/format.hpp>
#include <stdlib.h>
using namespace boost::filesystem;
using namespace std;


vector<string> splitString(const string& str,
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

	strings.push_back(str.substr(prev));
        
	return strings;
}

vector<vector<double>> getImageFeatures(string fName) {
   vector<vector<double>> instances;
   ifstream file(fName);
   string str;
   while (getline(file, str))
    {
       vector<double> features;
       vector<string> fs = splitString(str," ");
       for(int i=0;i < fs.size() ; i++) {
          if(fs.at(i) != "") {
             vector<string> fs1 = splitString(fs.at(i),":");
             features.push_back(stod(fs1.at(1)));

          }
       }
       instances.push_back(features);
    }
   return instances;


}
       
vector<vector<double>>  getInstances(string dsLoc,string inst,int cat) {
   string tLoc = dsLoc + "/" + inst;
   vector<string> instName = splitString(inst,"/");
   string catName = "";
   if(cat == 1) {
      catName = instName.at(1) + "_rgb.log";
   } else if(cat == 2) {
      catName = instName.at(1) + "_shape.log";
   } else if (cat == 3) {
      catName = instName.at(1) + "_object.log";
   }
   string fName = tLoc + "/" + catName;
   return getImageFeatures(fName);
}

double sigmoid(const vector<double>& data,const vector<double>& coef) {
  double yHat = coef[0];
  int x  = 4;
  for (int ij=1;ij < coef.size(); ij++) {
	yHat = yHat + coef[ij] * data[ij - 1];
  }
  double xx = -1 * yHat;
  double sig = 1.0/(1.0 + exp(xx));
  return sig;
}

double costFunction(vector<double> inst,vector<double> coef, double y) {
   double sig = sigmoid(inst,coef);
   double error = y - sig;
//   cout <<"sig " << sig << ", y---" << y <<", error -- " << error << ", ";
   double loss = error * sig * (1.0 - sig);
//   cout << "los ---" << loss << endl;
   return loss;
/*
   int x = 2;
   sig = floor( ( sig * pow( 10,x ) ) + 0.5 ) / pow( 10, x);
   error = floor( ( error * pow( 10,x ) ) + 0.5 ) / pow( 10, x);
  coef[0] = coef[0] + error * sig * (1.0 - sig);
  coef[0] = floor( ( coef[0] * pow( 10,x ) ) + 0.5 ) / pow( 10, x);
//   cout << sig << " " << error << " " << coef[0] << " ";
   for (int ij = 1; ij < coef.size(); ij++) {
	   coef[ij] = coef[ij] + error * sig * (1.0 - sig) * inst[ij-1];
	   coef[ij] = floor( ( coef[ij] * pow( 10,x ) ) + 0.5 ) / pow( 10, x);

   }
//   cout << coef[1] << endl;
   return coef;
*/
}
double costFunctionBatch(const vector<double>& inst,const vector<double>& coef, const double& y) {
   double sig = sigmoid(inst,coef);
   double error = y - sig;
   return error;
//   if(sig > 0.0 and sig < 1.0) {
       return error * sig * (1.0 - sig);
//   } else {
//       return error;
//   }
}

double regularize(vector<double> inst,vector<double> coef, double reg) {
   double eps = 0.1;

   double sig = sigmoid(inst,coef);  
   double dSig = sig * (1 - sig);
   int x = 2;
   dSig = floor( ( dSig * pow( 10,x ) ) + 0.5 ) / pow( 10, x);
   dSig = dSig - ( 1.0 - eps);
   dSig = dSig * dSig;
   reg = reg + dSig;
    return reg;
}

vector<double> updateRegularizer(vector<double> b,double g,double gG,int n) {
   double eta = 0.1;
   int x = 2;
//   g = floor( ( g * pow( 10,x ) ) + 0.5 ) / pow( 10, x);
//   gG = floor( ( gG * pow( 10,x ) ) + 0.5 ) / pow( 10, x);
   double s = eta * (g + (gG / n));
//   s = floor( ( s * pow( 10,x ) ) + 0.5 ) / pow( 10, x);
 //  cout << "s  " << s << endl;
   vector<double> bb;
   for(int i=0;i < b.size();i++) {
      double ss = b.at(i) + (s / b.size());
//      cout << g.at(i) << " ";
//      cout << s << " ";
      bb.push_back(ss);
 //     cout << ss << " ";
   }
 //  cout << endl;
   return bb;
}

vector<double> updateCostFunction(const vector<double>& inst,const vector<double>& b,const double& g,const double& gG,const int& n) {
   double eta = 0.1;
   int x = 2;
   double s = eta * (g + (gG / n));
   vector<double> bb;
   double ss = b.at(0) + s;
   bb.push_back(ss);
   for(int i=1;i < b.size();i++) {
      ss = b.at(i) + s * inst.at(i-1);
      bb.push_back(ss);
   }
   return bb;
}

vector<double> updateCostBatch(const vector<double>& b,const vector<double>& g,const double& gG,const int& n) {
   double eta = 0.1;
   vector<double> bb;
   double ss = b.at(0) + g.at(0) + eta * (gG / n);
//   cout <<  b.at(0) << "-" << g.at(0) << "-" << ss << endl;
   
   bb.push_back(ss);
//   cout <<  b.at(0) << "--" << ss << endl;
   for(int i=1;i < b.size();i++) {
      ss = b.at(i) + g.at(i) + eta * (gG / n);
      bb.push_back(ss);
//cout <<  b.at(i) << "--" << ss << endl;

   }
//   cout << endl << endl;
   return bb; 
}

vector<double> updateG(const vector<double>& inst,const vector<double>& b,const double& g) {
   double eta = 0.1;
   double s = g * eta;
   vector<double> bb;
   double ss = b.at(0) + s;
   bb.push_back(ss);
   for(int i=1;i < b.size();i++) {
      ss = b.at(i) + s * inst.at(i-1);
      bb.push_back(ss);
   }
   return bb;
}

double logLikelyHood(const vector<vector<double>>& instPos,const vector<vector<double>>& instNeg,const vector<double>& coef) {
   double ll = 0.0;
   for (int k = 0; k < instPos.size();k++) {
        double sig = sigmoid(instPos.at(k),coef);
        if(sig > 0.0) {
          ll += (1.0 * log(sig)) ;
	}
   }
   for (int k = 0; k < instNeg.size();k++) {
        double sig = sigmoid(instNeg.at(k),coef);
	if(sig < 1.0) {
           ll += (1.0 - 0.0) * log(1.0 - sig) ;
	}
   }
   return -1 * ll;
}

double negLossFunctionUpdate1(vector<double> coef,vector<string> dss,int type) {
   double loss = 0.0;
   for(int ij =0; ij < dss.size();ij++) {
      path p(dss.at(ij));
      for (auto i = directory_iterator(p); i != directory_iterator(); i++)
      {
        if (!is_directory(i->path())) 
	{
           string eDs = ".check";
	   if(type == 1) {
             eDs = "rgb.log";
	   } else if(type == 2) {
             eDs = "shape.log";
	   } else if (type == 3) {
             eDs = "object.log";
	   }
           if (boost::ends_with(i->path().filename(), eDs)) {
             ostringstream dte;
             dte << dss.at(ij) << "/" << i->path().filename();
	     vector<vector<double>> insts = getImageFeatures(dte.str());
             int r =  rand() % insts.size();
//             loss += costFunction(insts.at(r),coef,0.0);

             for(int k = 0;k < insts.size();k++) {
		loss +=  costFunction(insts.at(k),coef,0.0);		
	     }

	   }
        }
        else
            continue;
      }  
   }
   return loss;
}


vector<double> negLossFunctionUpdate(vector<double> coef,vector<string> dss,int type) {
   double loss = 0.0;
   for(int ij =0; ij < dss.size();ij++) {
      path p(dss.at(ij));
      for (auto i = directory_iterator(p); i != directory_iterator(); i++)
      {
        if (!is_directory(i->path())) 
        {
           string eDs = ".check";
           if(type == 1) {
             eDs = "rgb.log";
           } else if(type == 2) {
             eDs = "shape.log";
           } else if (type == 3) {
             eDs = "object.log";
           }
           if (boost::ends_with(i->path().filename(), eDs)) {
             ostringstream dte;
             dte << dss.at(ij) << "/" << i->path().filename();
	   //  cout << "----->" << dte.str() << endl;
             vector<vector<double>> insts = getImageFeatures(dte.str());
             int r =  rand() % insts.size();
 //            loss += costFunction(insts.at(r),coef,0.0);

             for(int k = 0;k < insts.size();k++) {
                loss =  costFunction(insts.at(k),coef,0.0);
		coef = updateRegularizer(coef,loss,0.0,type);
             }

           }
        }
        else
            continue;
      }
   }
   return coef;
}



vector<double> negCostUpdate(vector<double> coef,vector<string> dss,int type) {
   double loss = 0.0;
   for(int ij =0; ij < dss.size();ij++) {
      path p(dss.at(ij));
      for (auto i = directory_iterator(p); i != directory_iterator(); i++)
      {
        if (!is_directory(i->path()))
        {
           string eDs = ".check";
           if(type == 1) {
             eDs = "rgb.log";
           } else if(type == 2) {
             eDs = "shape.log";
           } else if (type == 3) {
             eDs = "object.log";
           }
           if (boost::ends_with(i->path().filename(), eDs)) {
             ostringstream dte;
             dte << dss.at(ij) << "/" << i->path().filename();
        //    cout << "----->" << dte.str() << endl;
             vector<vector<double>> insts = getImageFeatures(dte.str());
             int r =  rand() % insts.size();
             for(int k = 0;k < insts.size();k++) {
                loss =  costFunction(insts.at(k),coef,0.0);
                coef = updateCostFunction(insts.at(k),coef,loss,0.0,type);
        //        coef = updateRegularizer(coef,loss,0.0,type);
             }

           }
        }
        else
            continue;
      }
   }
   return coef;
}

void testLogRegression(string dsLoc,vector<string> testSet,map<string, vector<double>> bColor,map<string, vector<double>> bShape,map<string, vector<double>> bObject,string csvFile,map<string,string> testFullAnnotation) {
    map<string,vector<vector<double>>> cTestSets;
    map<string,vector<vector<double>>> sTestSets;
    map<string,vector<vector<double>>> oTestSets;
    string banner = ",";
    string tCases = ",";
    for(int i=0;i<testSet.size();i++) {
        vector<string> ar = splitString(testSet.at(i),"/");
        banner += "," + testFullAnnotation[ar.at(1)];
        tCases += "," + testSet.at(i);
        cTestSets[testSet.at(i)] = getInstances(dsLoc,testSet.at(i),1);
        sTestSets[testSet.at(i)] = getInstances(dsLoc,testSet.at(i),2);
        oTestSets[testSet.at(i)] = getInstances(dsLoc,testSet.at(i),3);       
    }
    ofstream myfile;
    myfile.open (csvFile);
    myfile << banner << "\n" << tCases << "\n";
    for (map<string,vector<double>>::iterator it=bColor.begin(); it!=bColor.end(); ++it) {
       string attribute = it->first;
       vector<double> cCoef = bColor[attribute];
       vector<double> sCoef = bShape[attribute];
       vector<double> oCoef = bObject[attribute];
       myfile << attribute + ",rgb";
       for(int i=0;i<testSet.size();i++) {
         vector<vector<double>> dZ = cTestSets[testSet.at(i)];
         double pred = 0.0;
         for(int j = 0;j < dZ.size();j++) {
		pred = pred + sigmoid(dZ.at(j),cCoef);		
	 }
//         cout << "---pred " << pred << " " << dZ.size() << endl;
         pred = pred / dZ.size();
         myfile << "," << pred;
       } 
       myfile << "\n";
       myfile << attribute + ",shape";
       for(int i=0;i<testSet.size();i++) {
         vector<vector<double>> dZ = sTestSets[testSet.at(i)];
         double pred = 0.0;
         for(int j = 0;j < dZ.size();j++) {
                double prec = sigmoid(dZ.at(j),sCoef);
                pred += prec;
         }
         pred = pred / dZ.size();
         myfile << "," << pred;
       }
       myfile << "\n";
       myfile << attribute + ",object";
       for(int i=0;i<testSet.size();i++) {
         vector<vector<double>> dZ = oTestSets[testSet.at(i)];
         double pred = 0.0;
         for(int j = 0;j < dZ.size();j++) {
                pred = pred + sigmoid(dZ.at(j),oCoef);
         }
         pred = pred / dZ.size();
         myfile << "," << pred;
       }
       myfile << "\n";
    }    
    cout << csvFile << endl;
    myfile.close();   

}

void regularizedLogisticRegression1(string dsLoc,vector<string> trainCls,vector<string> testSet,map<string,vector<string>> negMapInstances,map<string,string> testFullAnnotation,string trainLocation) {
      map<string, vector<double>> bColor;
      map<string, vector<double>> bShape;
      map<string, vector<double>> bObject;


      map<string, double> gColor;
      map<string, double> gShape;
      map<string, double> gObject;

      map<string, double> gGColor;
      map<string, double> gGShape;
      map<string, double> gGObject;
      map<string,int> instCount;
     
     for(int batch=0;batch<20;batch++) {
/// loop - description by decriptiona
      cout << "Batch --> " << batch << ", Training Set -- " << trainCls.size() << endl;
      for(int i = 0; i < trainCls.size(); i++) {
//	for(int i = 0; i < 3; i++) {
      //    cout << i << " -  " << trainCls.at(i) << endl;
          vector <string> feature = splitString(trainCls.at(i),",");
	  string annotation = feature.back();
	  vector<string> anntn = splitString(annotation," ");
	  vector<string> dss = negMapInstances[feature.front()];

          vector<vector<double>> rgbInsts = getInstances(dsLoc,feature.front(),1);
	  vector<vector<double>> shapeInsts = getInstances(dsLoc,feature.front(),2);
          vector<vector<double>> objInsts = getInstances(dsLoc,feature.front(),3);         
//loop - attributes for one instance
	  for(unsigned j = 0;j < anntn.size(); j++) {
             if(anntn.at(j) != "") {
		  string attribute = anntn.at(j);
/////beta / coefficient declaration
		  vector<double> rgbCoef;
		  vector<double> shapeCoef;
		  vector<double> objCoef;
		  if (bColor.find(attribute) == bColor.end() ) {
		     vector<double> coef(rgbInsts[0].size() + 1, 0);
		     rgbCoef = coef;	
		  } else {
			rgbCoef = bColor[attribute];
		  }
                  if (bShape.find(attribute) == bShape.end() ) {
                     vector<double> coef(shapeInsts[0].size() + 1, 0);
                     shapeCoef = coef;
                  } else {
                        shapeCoef = bShape[attribute];
                  }
                  if (bObject.find(attribute) == bObject.end() ) {
                     vector<double> coef(objInsts[0].size() + 1, 0);
                     objCoef = coef;
                  } else {
                        objCoef = bObject[attribute];
                  }
/// loss function gradient declaration
                  double cLoss = 0.0;
                  double sLoss = 0.0;
                  double oLoss = 0.0;
                  if(gColor.find(attribute) != gColor.end() ) {
                        cLoss = gColor[attribute];
                  }
                  if(gShape.find(attribute) != gShape.end() ) {
                        sLoss = gShape[attribute];
                  }
                  if(gObject.find(attribute) != gObject.end() ) {
                        oLoss = gObject[attribute];
                  }

//------regularizer declaration
		  double cReg = 0.0;
		  double sReg = 0.0;
		  double oReg = 0.0;
		  if(gGColor.find(attribute) != gGColor.end() ) {
			cReg = gGColor[attribute];
		  }
                  if(gGShape.find(attribute) != gGShape.end() ) {
                        sReg = gGShape[attribute];
                  }
                  if(gGObject.find(attribute) != gGObject.end() ) {
                        oReg = gGObject[attribute];
                  }

		  //loop --- images per instance
		  for (int k = 0; k < rgbInsts.size();k++) {
		// Regularizer
		    cReg = regularize(rgbInsts.at(k),rgbCoef,cReg);
                    sReg = regularize(shapeInsts.at(k),shapeCoef,sReg);
                    oReg = regularize(objInsts.at(k),objCoef,oReg);
		//
		// Loss function		     
		
                     cLoss += costFunction(rgbInsts.at(k),rgbCoef,1.0);
                     sLoss +=  costFunction(shapeInsts.at(k),shapeCoef,1.0);
                     oLoss +=  costFunction(objInsts.at(k),objCoef,1.0);
		     
		  }
                 
/*		  cLoss += negLossFunctionUpdate(rgbCoef,dss,1);
                  sLoss +=  negLossFunctionUpdate(shapeCoef,dss,2);
                  oLoss += negLossFunctionUpdate(objCoef,dss,3);
*/
		  bColor[attribute] = rgbCoef;
		  bShape[attribute] = shapeCoef;
		  bObject[attribute] = objCoef;

		  gGColor[attribute] = cReg;
		  gGShape[attribute] = sReg;
		  gGObject[attribute] = oReg;

                  gColor[attribute] = cLoss;
                  gShape[attribute] = sLoss;
                  gObject[attribute] = oLoss;

		  if(instCount.find(attribute) != instCount.end()){
			instCount[attribute] = instCount[attribute] + rgbInsts.size();
		  } else {
			instCount[attribute] = rgbInsts.size();
		  }
//			cout << dsLoc + "/" + feature.front() << endl;
	     }
	  }
      }
      for (map<string,int>::iterator it=instCount.begin(); it!=instCount.end(); ++it) {
	      int instC = it->second;
	      string attribute = it->first;
//	      cout << attribute << " --- ";
	      double g = gColor[attribute];
	      //vector<double> g = gColor[attribute];
              double gG = gGColor[attribute];
	      double gGCombined = gGColor[attribute] + gGShape[attribute] + gGObject[attribute];
//              cout << "g and g' " << g << " " << gG << "--" << instC << endl;
	      bColor[attribute] = updateRegularizer(bColor[attribute],gColor[attribute],gGCombined,instC);
              bShape[attribute] = updateRegularizer(bShape[attribute],gShape[attribute],gGCombined,instC);
              bObject[attribute] = updateRegularizer(bObject[attribute],gObject[attribute],gGCombined,instC);
      }
      gColor.clear();
      gShape.clear();
      gObject.clear();

      gGColor.clear();
      gGShape.clear();
      gGObject.clear();

      instCount.clear();	
      ostringstream dte;
      dte << trainLocation << "/" << (batch + 1);
      string cmd = "mkdir -p " + dte.str();
      int ret = system(cmd.c_str());
      string csvFile = dte.str() + "/regularizedResults.csv";
      testLogRegression(dsLoc,testSet,bColor,bShape,bObject,csvFile,testFullAnnotation);  
//      csvFile = trainLocation + "/traditionalResults.csv";   
//      testLogRegression(dsLoc,testSet,gColor,gShape,gObject,csvFile,testFullAnnotation);
}
}


void regularizedLogisticRegression2(string dsLoc,vector<string> trainCls,vector<string> testSet,map<string,vector<string>> negMapInstances,map<string,string> testFullAnnotation,string trainLocation) {
      map<string, vector<double>> bColor;
      map<string, vector<double>> bShape;
      map<string, vector<double>> bObject;


      map<string, double> gColor;
      map<string, double> gShape;
      map<string, double> gObject;

      map<string, double> gGColor;
      map<string, double> gGShape;
      map<string, double> gGObject;
      map<string,int> instCount;
 
      map<string, vector<string>> negPaths;  
     int batch;  
     for(batch=0;batch<5000;batch++) {
      cout << "Batch --> " << batch << ", Training Set -- " << trainCls.size() << endl;
/// loop - description by decription
      for(int i = 0; i < trainCls.size(); i++) {
//	for(int i = 0; i < 3; i++) {
//          cout << i << " -  " << trainCls.at(i) << endl;
          vector <string> feature = splitString(trainCls.at(i),",");
	  string annotation = feature.back();
	  vector<string> anntn = splitString(annotation," ");
	  vector<string> dss = negMapInstances[feature.front()];

          vector<vector<double>> rgbInsts = getInstances(dsLoc,feature.front(),1);
	  vector<vector<double>> shapeInsts = getInstances(dsLoc,feature.front(),2);
          vector<vector<double>> objInsts = getInstances(dsLoc,feature.front(),3);         
//loop - attributes for one instance
	  for(unsigned j = 0;j < anntn.size(); j++) {
             if(anntn.at(j) != "" and anntn.at(j) == "red") {
		  string attribute = anntn.at(j);
/////beta / coefficient declaration
		  vector<double> rgbCoef;
		  vector<double> shapeCoef;
		  vector<double> objCoef;
		  if (bColor.find(attribute) == bColor.end() ) {
		     vector<double> coef(rgbInsts[0].size() + 1, 0);
		     rgbCoef = coef;	
		  } else {
			rgbCoef = bColor[attribute];
		  }
                  if (bShape.find(attribute) == bShape.end() ) {
                     vector<double> coef(shapeInsts[0].size() + 1, 0);
                     shapeCoef = coef;
                  } else {
                        shapeCoef = bShape[attribute];
                  }

		  if (bObject.find(attribute) == bObject.end() ) {
                     vector<double> coef(objInsts[0].size() + 1, 0);
                     objCoef = coef;
                  } else {
                        objCoef = bObject[attribute];
                  }
	

                  if (negPaths.find(attribute) == negPaths.end() ) {
			negPaths[attribute] = dss;
                  } else {
			vector<string> a = negPaths[attribute];
			a.insert(std::end(a), std::begin(dss), std::end(dss));
			negPaths[attribute] = a;
                  }

/// loss function gradient declaration
                  double cLoss = 0.0;
                  double sLoss = 0.0;
                  double oLoss = 0.0;
                  if(gColor.find(attribute) != gColor.end() ) {
                        cLoss = gColor[attribute];
                  }
                  if(gShape.find(attribute) != gShape.end() ) {
                        sLoss = gShape[attribute];
                  }
                  if(gObject.find(attribute) != gObject.end() ) {
                        oLoss = gObject[attribute];
                  }

//------regularizer declaration
		  double cReg = 0.0;
		  double sReg = 0.0;
		  double oReg = 0.0;
		  if(gGColor.find(attribute) != gGColor.end() ) {
			cReg = gGColor[attribute];
		  }
                  if(gGShape.find(attribute) != gGShape.end() ) {
                        sReg = gGShape[attribute];
                  }
                  if(gGObject.find(attribute) != gGObject.end() ) {
                        oReg = gGObject[attribute];
                  }

		  //loop --- images per instance
		  for (int k = 0; k < rgbInsts.size();k++) {
		// Regularizer
		    cReg = regularize(rgbInsts.at(k),rgbCoef,0.0);
                    sReg = regularize(shapeInsts.at(k),shapeCoef,0.0);
                    oReg = regularize(objInsts.at(k),objCoef,0.0);
		//
		// Loss function		     
		
                     cLoss = costFunction(rgbInsts.at(k),rgbCoef,1.0);
                     sLoss =  costFunction(shapeInsts.at(k),shapeCoef,1.0);
                     oLoss =  costFunction(objInsts.at(k),objCoef,1.0);

	              double gG1 = cReg+ sReg + oReg;
		      gG1 = 0.0;
                      rgbCoef = updateCostFunction(rgbInsts.at(k),rgbCoef,cLoss,gG1,1);
                      shapeCoef = updateCostFunction(shapeInsts.at(k),shapeCoef,sLoss,gG1,2);
                      objCoef = updateCostFunction(objInsts.at(k),objCoef,oLoss,gG1,3);
		     
		  }
                
		  rgbCoef = negCostUpdate(rgbCoef,dss,1);
                  shapeCoef =  negCostUpdate(shapeCoef,dss,2);
                  objCoef = negCostUpdate(objCoef,dss,3);

/*	          double gG1 = 0.0;
                  rgbCoef = updateRegularizer(rgbCoef,cLoss,gG1,1);
                  shapeCoef = updateRegularizer(shapeCoef,sLoss,gG1,2);
                   objCoef = updateRegularizer(objCoef,oLoss,gG1,3);
*/

		  bColor[attribute] = rgbCoef;
		  bShape[attribute] = shapeCoef;
		  bObject[attribute] = objCoef;

		  gGColor[attribute] = cReg;
		  gGShape[attribute] = sReg;
		  gGObject[attribute] = oReg;

                  gColor[attribute] = cLoss;
                  gShape[attribute] = sLoss;
                  gObject[attribute] = oLoss;

		  if(instCount.find(attribute) != instCount.end()){
			instCount[attribute] = instCount[attribute] + rgbInsts.size();
		  } else {
			instCount[attribute] = rgbInsts.size();
		  }
//			cout << dsLoc + "/" + feature.front() << endl;
	     }
	  }
      }
/*
      for (map<string,vector<string>>::iterator it=negPaths.begin(); it!=negPaths.end(); ++it) {
              vector<string> vec = it->second;
              string attribute = it->first;
	      cout << "Negs ----> " << attribute << endl;
	      set<string> myset(vec.begin(), vec.end());
              vector<string> dss;
	      copy(myset.begin(), myset.end(), std::back_inserter(dss));
	      bColor[attribute] = negLossFunctionUpdate(bColor[attribute],dss,1);
	      bShape[attribute] =  negLossFunctionUpdate(bShape[attribute],dss,2);
	      bObject[attribute] = negLossFunctionUpdate(bObject[attribute],dss,3);	      	

      }
*/
      negPaths.clear();

      gColor.clear();
      gShape.clear();
      gObject.clear();

      gGColor.clear();
      gGShape.clear();
      gGObject.clear();

      instCount.clear();	
      }
      ostringstream dte;
      dte << trainLocation << "/" << batch;
      string cmd = "mkdir -p " + dte.str();
      int ret = system(cmd.c_str());
//      string csvFile = dte.str() + "/regularizedResults.csv";
      string csvFile = dte.str() + "/traditionalExecutionResults.csv";
      testLogRegression(dsLoc,testSet,bColor,bShape,bObject,csvFile,testFullAnnotation);  
//      csvFile = trainLocation + "/traditionalResults.csv";   
//      testLogRegression(dsLoc,testSet,gColor,gShape,gObject,csvFile,testFullAnnotation);
}

void regularizedLogisticRegression(string dsLoc,vector<string> trainCls,vector<string> testSet,map<string,vector<string>> negMapInstances,map<string,string> testFullAnnotation,string trainLocation) {
     int batch = 1000;
     map<string, vector<string>> posPaths;
     map<string, vector<string>> negPaths;
     map<string,vector<vector<double>>> rgbFeatures;
     map<string,vector<vector<double>>> shapeFeatures;
     map<string,vector<vector<double>>> objFeatures;
     for(int i = 0; i < trainCls.size(); i++) {
          vector <string> feature = splitString(trainCls.at(i),",");
          string annotation = feature.back();
          vector<string> anntn = splitString(annotation," ");
	  string instName = feature.front();
          vector<string> dss = negMapInstances[instName];

	  if (rgbFeatures.find(instName) == rgbFeatures.end() ) {
		  vector<vector<double>> insts = getInstances(dsLoc,instName,1);
		  rgbFeatures[instName] = insts;
		  insts = getInstances(dsLoc,instName,2);
		  shapeFeatures[instName] = insts;
		  insts = getInstances(dsLoc,instName,3);
		  objFeatures[instName] = insts;
	  }

          for(unsigned j = 0;j < anntn.size(); j++) {
             if(anntn.at(j) != "") {
                  string attribute = anntn.at(j);

		  if (posPaths.find(attribute) == posPaths.end() ) {
                         posPaths[attribute].push_back(instName);;
                  } else {
                        vector<string> a = posPaths[attribute];
			a.push_back(instName);
                        posPaths[attribute] = a;
                  }

                  if (negPaths.find(attribute) == negPaths.end() ) {
                        negPaths[attribute] = dss;
                  } else {
                        vector<string> a = negPaths[attribute];
                        a.insert(std::end(a), std::begin(dss), std::end(dss));
                        negPaths[attribute] = a;
                  }
	     }
	  }
     }
      map<string, vector<double>> bColor;
      map<string, vector<double>> bShape;
      map<string, vector<double>> bObject;
     for (map<string, vector<string>>::iterator it=posPaths.begin(); it!=posPaths.end(); ++it) {
              vector<string> posInstances = it->second;
              string attribute = it->first;
	      cout << "ATTRIBUTE ==> " << attribute << endl;
	      if (attribute != "red")
			continue;
	      vector<string> negInstances1 = negPaths[attribute];
	      set<string> s(negInstances1.begin(), negInstances1.end());
              vector<string> negInstances;
	      copy(s.begin(), s.end(), back_inserter(negInstances));
	      vector<vector<double>> posRgbFeatures;
              vector<vector<double>> posShapeFeatures;
              vector<vector<double>> posObjFeatures;

              vector<vector<double>> negRgbFeatures;
              vector<vector<double>> negShapeFeatures;
              vector<vector<double>> negObjFeatures;
	      for(int i=0;i < posInstances.size(); i++) {
		  string instName = posInstances.at(i);
		 // cout << instName << ", ";	  
		  if(posRgbFeatures.size() == 0) {
			posRgbFeatures = rgbFeatures[instName];
		  } else {
                        posRgbFeatures.insert(std::end(posRgbFeatures), std::begin(rgbFeatures[instName]), std::end(rgbFeatures[instName]));
		  }
                  if(posShapeFeatures.size() == 0) {
                        posShapeFeatures = shapeFeatures[instName];
                  } else {
                        posShapeFeatures.insert(std::end(posShapeFeatures), std::begin(shapeFeatures[instName]), std::end(shapeFeatures[instName]));     
                  }
                  if(posObjFeatures.size() == 0) {
                        posObjFeatures = objFeatures[instName];
                  } else {
                        posObjFeatures.insert(std::end(posObjFeatures), std::begin(objFeatures[instName]), std::end(objFeatures[instName]));     
                  }
	      }
            //  cout << endl << "Neg ---> ";
              for(int i=0;i < negInstances.size(); i++) {
                  string instName = negInstances.at(i);
             //     cout << instName << ", ";
                  if(negRgbFeatures.size() == 0) {
                        negRgbFeatures = rgbFeatures[instName];
                  } else {
                        negRgbFeatures.insert(std::end(negRgbFeatures), std::begin(rgbFeatures[instName]), std::end(rgbFeatures[instName]));
                  }
                  if(negShapeFeatures.size() == 0) {
                        negShapeFeatures = shapeFeatures[instName];
                  } else {
                        negShapeFeatures.insert(std::end(negShapeFeatures), std::begin(shapeFeatures[instName]), std::end(shapeFeatures[instName]));
                  }
                  if(negObjFeatures.size() == 0) {
                        negObjFeatures = objFeatures[instName];
                  } else {
                        negObjFeatures.insert(std::end(negObjFeatures), std::begin(objFeatures[instName]), std::end(objFeatures[instName]));
                  }
              }
//	      cout << attribute << "---" << posRgbFeatures.size() << "-" << posShapeFeatures.size() << "-" << posObjFeatures.size() << endl;	
//	      cout << attribute << "---" << negRgbFeatures.size() << "-" << negShapeFeatures.size() << "-" << negObjFeatures.size() << endl; 



//	    ////// Stochastic Updation /////////  
              cout << "----------Online Mode ----------" << endl;
	      vector<double> rgbCoef(posRgbFeatures[0].size() + 1, 0);
              vector<double> shapeCoef(posShapeFeatures[0].size() + 1, 0);
              vector<double> objCoef(posObjFeatures[0].size() + 1, 0);
	      for(int epochs = 0;epochs < batch; epochs++) {

	         for (int k = 0; k < posRgbFeatures.size();k++) {
		   // Regularizer
		   // Loss function  
		      double cLoss = costFunction(posRgbFeatures.at(k),rgbCoef,1.0);
		      double sLoss =  costFunction(posShapeFeatures.at(k),shapeCoef,1.0);
		      double oLoss =  costFunction(posObjFeatures.at(k),objCoef,1.0);

                      //double gG1 = cReg+ sReg + oReg;
                      double gG1 = 0.0;
                      rgbCoef = updateCostFunction(posRgbFeatures.at(k),rgbCoef,cLoss,gG1,1);
                      shapeCoef = updateCostFunction(posShapeFeatures.at(k),shapeCoef,sLoss,gG1,2);
                      objCoef = updateCostFunction(posObjFeatures.at(k),objCoef,oLoss,gG1,3);

		 }
		  
                 for (int k = 0; k < negRgbFeatures.size();k++) {
                   // Regularizer
                   //                    // Loss function  
                      double cLoss = costFunction(negRgbFeatures.at(k),rgbCoef,0.0);
                      double sLoss =  costFunction(negShapeFeatures.at(k),shapeCoef,0.0);
                      double oLoss =  costFunction(negObjFeatures.at(k),objCoef,0.0);

                      //double gG1 = cReg+ sReg + oReg;
                      double gG1 = 0.0;
                      rgbCoef = updateCostFunction(negRgbFeatures.at(k),rgbCoef,cLoss,gG1,1);
                      shapeCoef = updateCostFunction(negShapeFeatures.at(k),shapeCoef,sLoss,gG1,2);
                      objCoef = updateCostFunction(negObjFeatures.at(k),objCoef,oLoss,gG1,3);

                 }
                 int iter = 500;
                 if ((epochs + 1) % iter == 0) {
                         cout << attribute << "--- Epoch : " << epochs + 1 << endl;
                        double ll = logLikelyHood(posRgbFeatures,negRgbFeatures,rgbCoef);
                        cout << "Color :: Negative Log Likelihood :: " << ll << endl;
                        ll = logLikelyHood(posShapeFeatures,negShapeFeatures,shapeCoef);
                        cout << "Shape :: Negative Log Likelihood :: " << ll << endl;
                        ll = logLikelyHood(posObjFeatures,negObjFeatures,objCoef);
                        cout << "Object :: Negative Log Likelihood :: " << ll << endl;
                 }                                                            
	      }
	      bColor[attribute] = rgbCoef;
	      bShape[attribute] = shapeCoef;
	      bObject[attribute] = objCoef;




//////////////////// Batch Updation /////////
/*
              cout << "--------Batch Mode  starts ------------" << endl;
              vector<double> rgbCoefB(posRgbFeatures[0].size() + 1, 0);
              vector<double> shapeCoefB(posShapeFeatures[0].size() + 1, 0);
              vector<double> objCoefB(posObjFeatures[0].size() + 1, 0);

              for(int epochs = 0;epochs < batch; epochs++) {

                 vector<double> rgbG(posRgbFeatures[0].size() + 1, 0);
                 vector<double> shapeG(posShapeFeatures[0].size() + 1, 0);
                 vector<double> objG(posObjFeatures[0].size() + 1, 0);

//                 cout << attribute << "--- Epoch : " << epochs + 1 << endl;
                 for (int k = 0; k < posRgbFeatures.size();k++) {
 
                      double cLoss = costFunctionBatch(posRgbFeatures.at(k),rgbCoefB,1.0);
                      double sLoss =  costFunctionBatch(posShapeFeatures.at(k),shapeCoefB,1.0);
                      double oLoss =  costFunctionBatch(posObjFeatures.at(k),objCoefB,1.0);

                      double gG1 = 0.0;
                      rgbG = updateCostFunction(posRgbFeatures.at(k),rgbG,cLoss,gG1,1);
                      shapeG = updateCostFunction(posShapeFeatures.at(k),shapeG,sLoss,gG1,2);
                      objG = updateCostFunction(posObjFeatures.at(k),objG,oLoss,gG1,3);			

                 }

//		cout << "Negative starts here " << endl;
                 for (int k = 0; k < negRgbFeatures.size();k++) {
                   // Regularizer
                   //                    //                    // Loss function  
                   //

                      double cLoss = costFunctionBatch(negRgbFeatures.at(k),rgbCoefB,0.0);
                      double sLoss =  costFunctionBatch(negShapeFeatures.at(k),shapeCoefB,0.0);
                      double oLoss =  costFunctionBatch(negObjFeatures.at(k),objCoefB,0.0);

                      double gG1 = 0.0;
                      rgbG = updateCostFunction(negRgbFeatures.at(k),rgbG,cLoss,gG1,1);
                      shapeG = updateCostFunction(negShapeFeatures.at(k),shapeG,sLoss,gG1,2);
                      objG = updateCostFunction(negObjFeatures.at(k),objG,oLoss,gG1,3);
                 }

		 double gG1 = 0.0;

		 rgbCoefB = updateCostBatch(rgbCoefB,rgbG,gG1,posRgbFeatures.size());
		 shapeCoefB = updateCostBatch(shapeCoefB,shapeG,gG1,posRgbFeatures.size());
		 objCoefB = updateCostBatch(objCoefB,objG,gG1,posRgbFeatures.size());
		 int iter = 500;
		 if ((epochs == 0) or ((epochs + 1) % iter == 0)) {
			 cout << attribute << "--- Epoch : " << epochs + 1 << endl;
			double ll = logLikelyHood(posRgbFeatures,negRgbFeatures,rgbCoefB);
			cout << "Color :: Negative Log Likelihood :: " << ll << endl;
			ll = logLikelyHood(posShapeFeatures,negShapeFeatures,shapeCoefB);
                        cout << "Shape :: Negative Log Likelihood :: " << ll << endl;
			ll = logLikelyHood(posObjFeatures,negObjFeatures,objCoefB);
                        cout << "Object :: Negative Log Likelihood :: " << ll << endl;
		 }
		 rgbG.clear();
                 shapeG.clear();
                 objG.clear();


              }
	      ostringstream atr;
	      atr << attribute << "-batch";
              bColor[atr.str()] = rgbCoefB;
              bShape[atr.str()] = shapeCoefB;
              bObject[atr.str()] = objCoefB;                                                             
*/
     }
     ostringstream dte;
     dte << trainLocation << "/" << batch;
     string cmd = "mkdir -p " + dte.str();
     int ret = system(cmd.c_str());
//      string csvFile = dte.str() + "/regularizedResults.csv";
     string csvFile = dte.str() + "/traditionalExecutionResults.csv";
     testLogRegression(dsLoc,testSet,bColor,bShape,bObject,csvFile,testFullAnnotation);      
	

}
#endif
