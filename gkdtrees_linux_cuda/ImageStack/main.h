#ifndef MAIN_H
#define MAIN_H

#ifdef WIN32
#include <windows.h>
#include <float.h>
//#define isfinite _finite
#define popen _popen
#define pclose _pclose
#endif

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <ctype.h>
#include <map>
#include <string>
#include <vector>
#include <iostream>
//#include <unistd.h>


using namespace std;


// Below are the data structures and functions available to operations:
class Image;

Image &stack(size_t index);
void push(Image);
void pop();
void dup();
void pull(size_t);

int readInt(string);
float readFloat(string);
char readChar(string);
void parseCommands(vector<string>);

void start();
void end();

int randomInt(int min, int max);
float randomFloat(float min, float max);

// time since program start in seconds
float currentTime();

#include "macros.h"
#include "Exception.h"

// Here ends the data structures and functions available to operations

#include "Operation.h"
#include "Image.h"

// only the help operation really needs access to this one
extern map<string, Operation *> operationMap;
typedef map<string, Operation *>::iterator OperationMapIterator;

#endif
