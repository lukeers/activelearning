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

using namespace std;


// random generator function:
int myrandom (int i) { return std::rand()%i;}


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

vector<vector<double> > getDataSet(string trainSet,double level) {
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
                   d1.push_back(lbl1);
                   dataset.push_back(d1);
           }
        }
	return dataset;
}


void testLogRegression(string trainLocation,string label,vector<string> csvFiles,vector<string> category) {
   for(int ijk = 0; ijk < category.size();ijk++) {
	string id = category.at(ijk);
	string csvFile = csvFiles.at(ijk);
       string trainSet = trainLocation + "/" + label + "/" + id + "-trainSet.linear";
       string testSet = trainLocation + "/" + label + "/" + id + "-testSet.linear";
       double normalize = 1.0;
       if(id == "rgb") 
		normalize = 255.0;
       vector<vector<double> > dataset = getDataSet(trainSet,normalize);
       std::srand ( unsigned ( std::time(0) ) );	
 // using built-in random generator:
       std::random_shuffle ( dataset.begin(), dataset.end() );
       std::random_shuffle ( dataset.begin(), dataset.end(), myrandom);
       vector<vector<double> > tstSet = getDataSet(testSet, normalize);
       int n_epoch = 5000;
       double l_rate = 0.1;
       vector<double> coef = coefficients_sgd(dataset, l_rate, n_epoch);
       for(int ijk=0;ijk<coef.size() ; ijk++) {
//	       cout << coef[ijk] << ", ";
       }
       vector<double> predictions = testPredict(coef,tstSet);	

        ostringstream dte;
        dte << csvFile << "-" << n_epoch << "-" << l_rate << ".csv";
        string csF = dte.str();
        ostringstream dte1;
        dte1 << label;
        for(int ij=0;ij < predictions.size();ij++) {
           dte1 << "," << predictions.at(ij);
        }
        string cmd = "echo " + dte1.str() + " >> " + csF + ";";
        int ret = system(cmd.c_str());
       //test();
   }
}

#endif
