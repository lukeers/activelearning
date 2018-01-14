#include "lang.h"
#include <stdlib.h>
#include <sstream>
#include <iostream>
#include <string>
#include <stdio.h>
#include <fstream>
#include <boost/algorithm/string/classification.hpp>
#include <boost/algorithm/string/split.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/foreach.hpp>
#include <vector>

using namespace std;

string loc = "/home/npillai1/AL/Lang/";

vector<string>  processLanguage(string annotation) {
    string command ;
    command = "python " + loc + "langparse.py " + annotation ;
    int ret = system(command.c_str());
    command = loc + "temp.txt";
    ifstream fs(command.c_str(),ios::in);
    string line;
    getline(fs,line);
    typedef std::vector<std::string> Tokens;
    Tokens tokens;
    boost::split( tokens, line, boost::is_any_of(" ") );
//    cout << "Words :: ";
//    std::cout << tokens.size() << " tokens" << std::endl;
    BOOST_FOREACH( const std::string& i, tokens ) {
        std::cout << "'" << i << "'";
    }
    cout << endl;
    return tokens;

}

