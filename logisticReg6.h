#ifndef LOGISTICREG_H
#define LOGISTICREG_H

#include <fstream>
#include <vector>
#include <iostream>
#include <math.h>
#include <sstream>
#include <cmath>
#include <algorithm>
#include <ctime>        // std::time
#include <cstdlib>      // std::rand, std::srand
#include <map>

using namespace std;

double regularizer;

// random generator function:
int myrandom (int i) { return std::rand()%i;}
map<string,vector<vector<double> >> trainingSet;

vector<string> split(const string& str,
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

double predict(vector<double> data,vector<double> coef) {
  double yHat = coef[0];
  for (int ij=0;ij < data.size() - 1; ij++) {
     yHat += coef[ij+1] * data[ij];
  }
//  yHat += regularizer;
  return 1.0/(1.0 + exp(-1 * yHat));

}

vector<double> coefficients_sgd(vector < vector<double> > dataset, double l_rate, int n_epoch) {
 vector<double> coef(dataset[0].size(), 0);
 for(int i = 0; i < n_epoch; i++) {
   double sum_error = 0.0;
   for(int j = 0; j < dataset.size();j++) {
      double yHat = predict(dataset[j],coef); 
      double error = dataset[j][dataset[j].size() - 1] - yHat;
       sum_error = sum_error + error * error;
      coef[0] = coef[0] + l_rate * error * yHat * (1.0 - yHat);
      for (int ij=0;ij < dataset[j].size() - 1; ij++) {
         coef[ij + 1] = coef[ij + 1] + l_rate * error * yHat * (1.0 - yHat) * dataset[j][ij];
      }
 //     cout << "error " << j << " instance : " << error << endl;
   }
//   cout << "epoch " << i << ", lrate " << l_rate << ", sum of error " << sum_error << endl;
 }
 return coef;
}

vector<double> testPredict(vector<double> coef,vector<vector<double> > testSet) {
   vector<double> predictions;
   for(int j = 0; j < testSet.size();j++) {
	double pred = predict(testSet[j],coef);
 	predictions.push_back(pred);
        vector<double> tst = testSet[j];
        
//	cout << pred << " " << tst.at(tst.size() - 1) << endl;
   }
   return predictions;

}

int test()
{
   vector<vector<double> > dataset {{2.7810836,2.550537003,0.0},
        {1.465489372,2.362125076,0},
        {3.396561688,4.400293529,0},
        {1.38807019,1.850220317,0},
        {3.06407232,3.005305973,0},
        {7.627531214,2.759262235,1},
        {5.332441248,2.088626775,1},
        {6.922596716,1.77106367,1},
        {8.675418651,-0.242068655,1},
        {7.673756466,3.508563011,1}};

   double l_rate = 0.3;
   int n_epoch = 100;
   vector<double> coef = coefficients_sgd(dataset, l_rate, n_epoch);
   for(int ijk=0;ijk<coef.size() ; ijk++) {
      cout << coef[ijk] << ", ";
   }
   cout << endl;
   vector<vector<double> > testSet {{3.7810836,1.550537003,0.0},
        {4.38807019,3.850220317,0},
        {6.332441248,7.088626775,1},
        {5.673756466,1.508563011,1}};
   vector<double> predictions = testPredict(coef,testSet);  
   
   return 0;
}

vector<vector<double> > getDataSet(string trainSet,double level,double val) {
        ifstream file(trainSet);
        string str;
        vector<vector<double> > dataset;
        while (getline(file, str)) {
           if (str != "") {
                   vector <string> elements = split(str," ");
                   vector<double> d1;
                   double lbl = stof(elements.at(0));
                   for (int i = 1;i < elements.size();i++) {
                        string elem = elements.at(i);
                        if (elem != "" ) {
                                vector <string> elems2 = split(elem,":");
                                double feature = stof(elems2.at(1)) / level;
                                d1.push_back(feature);
                        }

                   }
                   double lbl1 = lbl;
                   if(lbl < 0.0) {
                        lbl1 = 0.0;
                   } else {
                        lbl1 = 1.0;
                   }
		   if (val == lbl1 or val == 2.0) {
			   d1.push_back(lbl1);
			   dataset.push_back(d1);
		   }
           }
        }
	return dataset;
}

vector<vector<double> > preprocessTrainData(string id,string trainLocation,string label,double normalize,double val) {
       string trainSet = trainLocation + "/" + label + "/" + id + "-trainSet.linear";
       vector<vector<double> > dataset = getDataSet(trainSet,normalize,val);
/*
       std::srand ( unsigned ( std::time(0) ) );
       std::random_shuffle ( dataset.begin(), dataset.end() );
       std::random_shuffle ( dataset.begin(), dataset.end(), myrandom);
*/
       return dataset;
}


int getThePositiveImagesCount(map<string,vector<vector<double> >> posSet,map<string,vector<vector<double> >> negSet){
    map<string,vector<vector<double> >> ::iterator itr;
    int mPCount = 0;
    int mNCount = 0;
    vector<int> posNo;
    vector<int> negNo;
    map<string,vector<vector<double> >> tnSet;
    for (itr = posSet.begin(); itr != posSet.end(); ++itr)
    {   
        int pCount = 0;
        int nCount = 0;
        vector<vector<double> > itC = itr->second;
	string id = itr->first;
	if(itC.size() > mPCount) {
		mPCount = itC.size();
	}
	posNo.push_back(itC.size());

	itC = negSet[id];
	if(itC.size() > mNCount) {
                mNCount = itC.size();
        }
        negNo.push_back(itC.size());
    }


/*
    for (itr = trainingSet.begin(); itr != trainingSet.end(); ++itr)
    {
	int pCount = 0;
	int nCount = 0;
        vector<vector<double> > itC = itr->second;
	for(int it = 0; it < itC.size(); it++) {
            vector<double> itV = itC.at(it);
	    if(itV.at(itV.size() - 1) == 1.0) {
               pCount += 1;
	    } else {
               nCount += 1;
	    }		
	}
	if(pCount > mPCount) {
		mPCount = pCount; 
	}
	if(nCount > mNCount) {
		mNCount = nCount;
	}
	posNo.push_back(pCount);
	negNo.push_back(nCount);
//	cout << pCount << "-" << nCount << "=" << mPCount << "-" << mNCount << endl;
    }
*/
   
    int it = 0;
    map<string,vector<vector<double> >> ::iterator itrB;
    for (itrB = posSet.begin(); itrB != posSet.end(); ++itrB)
    {
            vector<vector<double> > itC = itrB->second;
	    string id = itrB->first;
            int pNo = posNo.at(it);
	    vector<vector<double> > itC1 = itC;
	    int chIndex = 0;
	    while(pNo < mPCount) {
		    int index = chIndex % itC.size();
		    vector<double> inst = itC.at(index);	
		    itC1.push_back(inst);
		    pNo += 1;
	    }    
	    posSet[id] = itC1;	
	    chIndex = 0;
            int nNo = negNo.at(it);
	    itC = negSet[id];
	    itC1 = itC;
	    while(nNo < mNCount ) {
                int index = chIndex % itC.size();
                vector<double> inst = itC.at(index); 
		itC1.push_back(inst);
		nNo += 1;
	    }
	    negSet[id] = itC1;
	    it += 1;
    }
    vector<int> indices;
    for(int ij = 0; ij < (mPCount + mNCount); ij++) {
       indices.push_back(ij);
    }
    trainingSet.clear();
    std::srand ( unsigned ( std::time(0) ) );
    std::random_shuffle ( indices.begin(), indices.end() );
    std::random_shuffle ( indices.begin(), indices.end(), myrandom);
    for (itrB = posSet.begin(); itrB != posSet.end(); ++itrB)
    { 
        vector<vector<double> > itC = itrB->second;
	string id = itrB->first;  
	vector<vector<double> > itNeg = negSet[id];
        vector<vector<double> > ds;
	for(int ij = 0; ij < indices.size(); ij++) {
		int ind = indices.at(ij);
		if(ind < mPCount) {
			ds.push_back(itC.at(ind));
		} else {
			ind = ind - mPCount;
			ds.push_back(itNeg.at(ind));
		}
	}
	trainingSet[id] = ds;
    }   
/*
    int it = 0;
    map<string,vector<vector<double> >> ::iterator itrB;
    for (itrB = posSet.begin(); itrB != posSet.end(); ++itrB)
    {
	    vector<vector<double> > itC = itrB->second;
	    int pNo = posNo.at(it);
	    int nNo = negNo.at(it);
	    int chIndex = 0;
	    vector<vector<double> > itC1 = itC;
	    while(pNo < mPCount) {
		int index = chIndex % itC.size();
		vector<double> inst = itC.at(index);
		if(inst.at(inst.size() - 1) == 1.0) {
			itC1.push_back(inst);
			pNo += 1;
		}
		chIndex += 1; 	
	    }
            chIndex = 0;
	    while(nNo < mNCount ) {
	        int index = chIndex % itC.size();
                vector<double> inst = itC.at(index);  
                if(inst.at(inst.size() - 1) == 0.0) {
                        itC1.push_back(inst);
                        nNo += 1;
                }
                chIndex += 1;	
	    }
	    string id = itrB->first;
	    tnSet[id] = itC1;  
	    it += 1;
    }

    trainingSet.clear();
    trainingSet = tnSet;
*/
    return mPCount;

}


map<string,vector<double>> sgdRegularized(double l_rate, int n_epoch, int x) {
  regularizer = 0.0;
  double eps = 0.01;
  map<string,vector<double>> coefs;
  map<string,vector<vector<double> >> ::iterator itrB;
  int D = 0;
  vector<vector<double> > rgbTrain;
  vector<vector<double> > shapeTrain;
  vector<vector<double> > objTrain;
  vector<double> rgbCoef;
  vector<double> shapeCoef;
  vector<double> objCoef;
  for (itrB = trainingSet.begin(); itrB != trainingSet.end(); ++itrB)
  {
            vector<vector<double> > itC = itrB->second;
  	    D = itC.size();
            string id = itrB->first;
	    vector<double> coef(itC[0].size(), 0);
	    if(id == "rgb") {
                rgbTrain = itC;
		rgbCoef = coef;
	    } else if(id == "shape") {
                shapeTrain = itC;
		shapeCoef = coef;
            } else {
                objTrain = itC;
		objCoef = coef;
            } 
  }
  double fAtr = 0.0;  
  int pCn = 0; 
  for(int j = 0; j < D; j++) {
	  vector<double> ds1 = rgbTrain.at(j);
	  vector<double> ds2 = shapeTrain.at(j);
	  vector<double> ds3 = objTrain.at(j);
	  double y1 = ds1.at(ds1.size() - 1);
	  double y2 = ds2.at(ds2.size() - 1);
	  double y3 = ds3.at(ds3.size() - 1);
	  if(y1 == 1.0 and y2 == 1.0 and y3 == 1.0) {
                pCn += 1;
	  }	  

  }
  for(int i = 0; i < n_epoch; i++) {
      double rReg = 0.0 ;
      double sReg = 0.0;
      double oReg = 0.0;
      for(int j = 0; j < D; j++) {
	vector<double> ds1 = rgbTrain.at(j);
        vector<double> ds2 = shapeTrain.at(j);
        vector<double> ds3 = objTrain.at(j);
	
	double yHat1 = predict(ds1,rgbCoef);
	double y1 = ds1.at(ds1.size() - 1);
        double error = y1 - yHat1;	

	double dSigmoid = yHat1 * (1.0 - yHat1);

	vector<double> rCoef(rgbCoef.size(), 0);
	rCoef[0] = error * dSigmoid;
	for (int ij=0;ij < rCoef.size() - 1; ij++) {
           rCoef[ij + 1] = rCoef[ij + 1] + error * dSigmoid * ds1[ij];
        }
/*
        rgbCoef[0] = rgbCoef[0] + l_rate * error * yHat1 * (1.0 - yHat1);
	
 * store error * yHat{i} * (1.0 - yHat{i}) * ds{i}[ij] in gradient vectors
 * 	
	for (int ij=0;ij < ds1.size() - 1; ij++) {
           rgbCoef[ij + 1] = rgbCoef[ij + 1] + l_rate * error * yHat1 * (1.0 - yHat1) * ds1[ij];
	}
*/
	
        double yHat2 = predict(ds2,shapeCoef);
        double y2 = ds2.at(ds2.size() - 1);
        error = y2 - yHat2;

        double dSigmoid2 = yHat2 * (1.0 - yHat2);

        vector<double> sCoef(shapeCoef.size(), 0);
        sCoef[0] = error * dSigmoid2;
        for (int ij=0;ij < sCoef.size() - 1; ij++) {
           sCoef[ij + 1] = sCoef[ij + 1] + error * dSigmoid2 * ds1[ij];
        }
/*
        shapeCoef[0] = shapeCoef[0] + l_rate * error * yHat2 * (1.0 - yHat2);
        for (int ij=0;ij < ds2.size() - 1; ij++) {
           shapeCoef[ij + 1] = shapeCoef[ij + 1] + l_rate * error * yHat2 * (1.0 - yHat2) * ds2[ij];
        }
*/
        double yHat3 = predict(ds3,objCoef);
        double y3 = ds3.at(ds3.size() - 1);
        error = y3 - yHat3;

        double dSigmoid3 = yHat3 * (1.0 - yHat3);

        vector<double> oCoef(objCoef.size(), 0);
        oCoef[0] = error * dSigmoid3;
        for (int ij=0;ij < oCoef.size() - 1; ij++) {
           oCoef[ij + 1] = oCoef[ij + 1] + error * dSigmoid3 * ds1[ij];
        }

/*
        objCoef[0] = objCoef[0] + l_rate * error * yHat3 * (1.0 - yHat3);
        for (int ij=0;ij < ds3.size() - 1; ij++) {
           objCoef[ij + 1] = objCoef[ij + 1] + l_rate * error * yHat3 * (1.0 - yHat3) * ds3[ij];
        }
*/
	if(y1 == 1.0 and y2 == 1.0 and y3 == 1.0) {
/*
 * add partial[sigmoid_c]/partial[*Coef] to the appropriate gradient vectors:
 */	
	rReg += yHat1 - ( 1.00 + eps);
		sReg += yHat2 - ( 1.00 + eps);
		oReg += yHat3 - ( 1.00 + eps);
/*
		fAtr = yHat1 - ( 1.00 + eps) + yHat2 - ( 1.00 + eps) + yHat3 - ( 1.00 + eps);
		for (int ij=0;ij < ds1.size() - 1; ij++) {
			rgbCoef[ij + 1] = rgbCoef[ij + 1] + fAtr/((ds1.size() - 1) * pCn);
		}
		for (int ij=0;ij < ds2.size() - 1; ij++) {
			shapeCoef[ij + 1] = shapeCoef[ij + 1] + fAtr/((ds2.size() - 1) * pCn);
		}
		for (int ij=0;ij < ds3.size() - 1; ij++) {
			objCoef[ij + 1] = objCoef[ij + 1] + fAtr/((ds3.size() - 1) * pCn);
		}

*/
//		fAtr += yHat1 - ( 1.00 + eps) + yHat2 - ( 1.00 + eps) + yHat3 - ( 1.00 + eps);
//		pCn += 1;
	}


/*
	   int pCn = 0;	 
	   for (itPtr = trainingSet.begin(); itPtr != trainingSet.end(); ++itPtr)
	   {
		string id = itPtr->first;
		vector<double> coef = coefs[id];
 		vector<vector<double> > itC = itPtr->second;
		vector<double> ds = itC.at(j);
		double yHat = predict(ds,coef);
		double y = ds.at(ds.size() - 1);
		double error = y - yHat;
		coef[0] = coef[0] + l_rate * error * yHat * (1.0 - yHat);
		for (int ij=0;ij < ds.size() - 1; ij++) {
			coef[ij + 1] = coef[ij + 1] + l_rate * error * yHat * (1.0 - yHat) * ds[ij];
		}
		coefs[id] = coef;
		if(y == 1.0) {
			pCn += 1;
			fAtr += yHat - ( 1.00 + eps);
		}
	   }
	   if(pCn == trainingSet.size()) {
		for (itPtr = trainingSet.begin(); itPtr != trainingSet.end(); ++itPtr)
		{    
			string id = itPtr->first;
			vector<double> coef = coefs[id];

			for (int ij=1;ij < coef.size(); ij++) {
				coef[ij] = coef[ij] + fAtr / ((coef.size() -1 ) * x);
			}
			coefs[id] = coef;
		}		
	   }	   	   
*/  

      }
      for (int ij=0;ij < rgbCoef.size() - 1; ij++) {
              rgbCoef[ij + 1] = rgbCoef[ij + 1] + rReg/((rgbCoef.size() - 1) * pCn);
      } 
      for (int ij=0;ij < shapeCoef.size() - 1; ij++) {
              shapeCoef[ij + 1] = shapeCoef[ij + 1] + sReg/((shapeCoef.size() - 1) * pCn);
      } 
      for (int ij=0;ij < objCoef.size() - 1; ij++) {
	      objCoef[ij + 1] = objCoef[ij + 1] + oReg/((objCoef.size() - 1) * pCn);
      }
      //        //   cout << "epoch " << i << ", lrate " << l_rate << ", sum of error " << sum_error << endl;
    }
/*
  for (int ij=1;ij < rgbCoef.size(); ij++) {
     cout << rgbCoef[ij] ;
     rgbCoef[ij] = rgbCoef[ij] + fAtr / ((rgbCoef.size() -1 ) * pCn);
     cout << " " << rgbCoef[ij] << endl;
  }
  for (int ij=1;ij < shapeCoef.size(); ij++) {
     shapeCoef[ij] = shapeCoef[ij] + fAtr / ((shapeCoef.size() -1 ) * pCn);
  }                     
  for (int ij=1;ij < objCoef.size(); ij++) {
     objCoef[ij] = objCoef[ij] + fAtr / ((objCoef.size() -1 ) * pCn);
  }
*/   
  coefs["rgb"] = rgbCoef;
  coefs["shape"] = shapeCoef;                  
  coefs["object"] = objCoef;

  //regularizer = fAtr / pCn;
  return coefs;
}

map<string,vector<double>> sgdLogisticRegression(double l_rate, int n_epoch) {
	map<string,vector<double>> coefs;
	map<string,vector<vector<double> >> ::iterator itrB;
	for (itrB = trainingSet.begin(); itrB != trainingSet.end(); ++itrB)
	{       
		vector<vector<double> > dataset = itrB->second;
		string id = itrB->first;
		vector<double> coef(dataset.size(), 0);
		for(int i = 0; i < n_epoch; i++) {
			for(int j = 0; j < dataset.size();j++) {
				double yHat = predict(dataset[j],coef);
				double error = dataset[j][dataset[j].size() - 1] - yHat;
				coef[0] = coef[0] + l_rate * error * yHat * (1.0 - yHat);
				for (int ij=0;ij < dataset[j].size() - 1; ij++) {
					coef[ij + 1] = coef[ij + 1] + l_rate * error * yHat * (1.0 - yHat) * dataset[j][ij];
				}
			}
		}
		coefs[id] = coef;
	}
	return coefs;
}


void testLogRegression(string trainLocation,string label,vector<string> csvFiles,vector<string> category) {
   trainingSet.clear();
   map<string,vector<vector<double> >> testingSet;
   map<string,vector<vector<double> >> posSet;
   map<string,vector<vector<double> >> negSet;
   int posM = 10000;
   int negM = 10000;
   int testD = 0;
   for(int ijk = 0; ijk < category.size();ijk++) {
	string id = category.at(ijk);
	string csvFile = csvFiles.at(ijk);
	double normalize = 1.0;
	if(id == "rgb")
                normalize = 255.0;
        vector<vector<double> > dataset = preprocessTrainData(id,trainLocation,label,normalize,1.0);
	posSet[id] = dataset;
	if(dataset.size() < posM) {
		posM = dataset.size();
	}
        vector<vector<double> > negdataset = preprocessTrainData(id,trainLocation,label,normalize,0.0);
	negSet[id] = negdataset;
	if(negdataset.size() < negM) {
                negM = negdataset.size();
        }
//	trainingSet[id] = dataset;
	string testSet = trainLocation + "/" + label + "/" + id + "-testSet.linear";
	vector<vector<double> > tstSet = getDataSet(testSet, normalize,2.0);
	testD = tstSet.size();
	testingSet[id] = tstSet;
   }
   int x = getThePositiveImagesCount(posSet,negSet);
   cout << testingSet.size() << endl;
   int n_epoch = 5000;
   double l_rate = 0.1;
 
   map<string,vector<double>> coefs;
   map<string,vector<vector<double> >> ::iterator itrB;
   int D = 0;
   vector<vector<double> > rgbTrain;
   vector<vector<double> > shapeTrain;
   vector<vector<double> > objTrain;
   vector<double> rgbCoef;
   vector<double> shapeCoef;
   vector<double> objCoef;
   vector<double> rgbCoef1;
   vector<double> shapeCoef1;
   vector<double> objCoef1;
   for (itrB = trainingSet.begin(); itrB != trainingSet.end(); ++itrB)
   {
            vector<vector<double> > itC = itrB->second;
            D = itC.size();
            string id = itrB->first;
            vector<double> coef(itC[0].size(), 0);
            if(id == "rgb") {
                rgbTrain = itC;
                rgbCoef = coef;
                rgbCoef1 = coef;
            } else if(id == "shape") {
                shapeTrain = itC;
                shapeCoef = coef;
                shapeCoef1 = coef;
            } else {
                objTrain = itC;
                objCoef = coef;
                objCoef1 = coef;
            }
   }
   for(int i = 0; i < n_epoch; i++) { 
	   double eps = 0.1;
	   double fattr = 0.0;
	   int nPn = 0;
	   for(int j = 0; j < D;j++) {
		   vector<vector<double> > dataset = rgbTrain;
		   vector<double> coef = rgbCoef;
		   double yHat = predict(dataset[j],coef);
		   double y = dataset[j][dataset[j].size() - 1];
		   double error = y - yHat;
		   coef[0] = coef[0] + l_rate * error * yHat * (1.0 - yHat);
		   for (int ij=0;ij < dataset[j].size() - 1; ij++) {
			   coef[ij + 1] = coef[ij + 1] + l_rate * error * yHat * (1.0 - yHat) * dataset[j][ij];
		   }
		   rgbCoef = coef;
/* 
 * * store error * yHat{i} * (1.0 - yHat{i}) * ds{i}[ij] in gradient vectors
 *  *  * 
 *   */
/***2nd regularized model ***/
		   double dSigmoid = yHat * (1.0 - yHat);

		   vector<double> rCoef(rgbCoef.size(), 0);
		   rCoef[0] = error * dSigmoid;
		   for (int ij=0;ij < dataset[j].size() - 1; ij++) {
			   rCoef[ij + 1] = error * dSigmoid * dataset[j][ij];
		   }

/*
                   rgbCoef1[0] += l_rate * error * yHat * (1.0 - yHat);
                   for (int ij=0;ij < dataset[j].size() - 1; ij++) {
                           rgbCoef1[ij + 1] += l_rate * error * yHat * (1.0 - yHat) * dataset[j][ij];
                   }
*/
                   dataset = shapeTrain;
                   coef = shapeCoef;
                   double yHat1 = predict(dataset[j],coef);
		   double y1 = dataset[j][dataset[j].size() - 1];
                   error = y1 - yHat1;
                   coef[0] = coef[0] + l_rate * error * yHat1 * (1.0 - yHat1);
                   for (int ij=0;ij < dataset[j].size() - 1; ij++) {
                           coef[ij + 1] = coef[ij + 1] + l_rate * error * yHat1 * (1.0 - yHat1) * dataset[j][ij];
                   }
                   shapeCoef = coef;

/***regularized model */
		   double dSigmoid2 = yHat1 * (1.0 - yHat1);

		   vector<double> sCoef(shapeCoef.size(), 0);
		   sCoef[0] = error * dSigmoid2;
		   for (int ij=0;ij < dataset[j].size() - 1; ij++) {
			   sCoef[ij + 1] =  error * dSigmoid2 * dataset[j][ij];
		   }


/*                   shapeCoef1[0] += l_rate * error * yHat * (1.0 - yHat);
                   for (int ij=0;ij < dataset[j].size() - 1; ij++) {
                           shapeCoef1[ij + 1] += l_rate * error * yHat * (1.0 - yHat) * dataset[j][ij];
                   }
*/
                   dataset = objTrain;
                   coef = objCoef;
                   double yHat2 = predict(dataset[j],coef);
		   double y2 = dataset[j][dataset[j].size() - 1];
                   error = y2 - yHat2;
                   coef[0] = coef[0] + l_rate * error * yHat2 * (1.0 - yHat2);
                   for (int ij=0;ij < dataset[j].size() - 1; ij++) {
                           coef[ij + 1] = coef[ij + 1] + l_rate * error * yHat2 * (1.0 - yHat2) * dataset[j][ij];
                   }
                   objCoef = coef;


/***regularized model */

		   double dSigmoid3 = yHat2 * (1.0 - yHat2);

		   vector<double> oCoef(objCoef.size(), 0);
		   oCoef[0] = error * dSigmoid3;
		   for (int ij=0;ij < dataset[j].size() - 1; ij++) {
			   oCoef[ij + 1] = error * dSigmoid3 * dataset[j][ij];
		   }

/*                   objCoef1[0] += l_rate * error * yHat * (1.0 - yHat);
                   for (int ij=0;ij < dataset[j].size() - 1; ij++) {
                           objCoef1[ij + 1] += l_rate * error * yHat * (1.0 - yHat) * dataset[j][ij];
                   }
*/
		   if(y == 1.0 and y1 == 1.0 and y2 == 1.0) {
/*
 *  * add partial[sigmoid_c]/partial[*Coef] to the appropriate gradient vectors:
 *   */

			   for (int ij=0;ij < rgbCoef1.size(); ij++) {
				   rgbCoef1[ij] = rgbCoef1[ij] + rCoef[ij] ;
			   }       
			   for (int ij=0;ij < shapeCoef1.size(); ij++) { 
				   shapeCoef1[ij] = shapeCoef1[ij] + sCoef[ij];
			   }       
			   for (int ij=0;ij < objCoef1.size(); ij++) {
				   objCoef1[ij] = objCoef1[ij] + oCoef[ij];
			   } 
/*
				fattr += yHat - ( 1 + eps) + yHat1 - ( 1 + eps) + yHat2 - ( 1 + eps) ;
				nPn += 1;
*/		   }
	   }
/*
	   for (int ij=0;ij < rgbCoef1.size() - 1; ij++) {
              rgbCoef1[ij + 1] = rgbCoef1[ij + 1] + fattr/((rgbCoef1.size() - 1) * nPn);
	   }
           for (int ij=0;ij < shapeCoef1.size() - 1; ij++) {
              shapeCoef1[ij + 1] = shapeCoef1[ij + 1] + fattr/(( shapeCoef1.size() - 1) * nPn);
           }
           for (int ij=0;ij < objCoef1.size() - 1; ij++) {
              objCoef1[ij + 1] = objCoef1[ij + 1] + fattr/((objCoef1.size() - 1) * nPn);
           }	  	   
*/  
   }

   string csvFile = csvFiles.at(0);
  
   string csvFile1 = csvFiles.at(1);

   for(int ijk = 0; ijk < category.size();ijk++) {
       
	   stringstream dte1;
	   dte1 << label;

	   string csvFile1 = csvFiles.at(1);
	   ostringstream dte2;
	   dte2 << label;
	   string id = category.at(ijk);
	dte1 << "," << id;
        dte2 << "," << id;
	vector<double> coef,coef1;
	if(id == "rgb") {
		coef = rgbCoef;
                coef1 = rgbCoef1;
	} else if(id == "shape") {
                coef  = shapeCoef;
                coef1  = shapeCoef1;
	} else {
                coef  = objCoef ;
                coef1  = objCoef1 ;
	}

        vector<vector<double> > tstSet = testingSet[id];
	vector<double> predictions = testPredict(coef,tstSet);	
	for(int ij=0;ij < predictions.size();ij++) {
           dte1 << "," << predictions.at(ij);
        }
	predictions = testPredict(coef1,tstSet);
        for(int ij=0;ij < predictions.size();ij++) {
           dte2 << "," << predictions.at(ij);
        }
	string cmd = "echo " + dte1.str() + " >> " + csvFile + ";";
	cout << cmd << endl;
	int ret = system(cmd.c_str());
	cmd = "echo " + dte2.str() + " >> " + csvFile1 + ";";
	cout << cmd << endl;
	ret = system(cmd.c_str());
   }
   cout << "----DONE-----" << label << endl;
}

#endif
