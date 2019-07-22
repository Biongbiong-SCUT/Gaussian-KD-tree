#ifndef IMAGESTACK_H
#define IMAGESTACK_H

// We never want SDL when used as a library
#define NO_SDL

// includes that don't survive well in the namespace
#include <string>
#include <vector>
#include <map>
#include <iostream>
#include <sstream>
#include <fstream>

#ifdef WIN32
#include <winsock2.h>
#else
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <netdb.h>
#endif

#include <math.h>

namespace ImageStack {

#include "main.h"
#include "Operation.h"
#include "Calculus.h"
#include "Color.h"
#include "Control.h"
#include "Convolve.h"
#include "DFT.h"
#include "Display.h"
#include "DisplayWindow.h"
#include "Dual.h"
#include "Exception.h"
#include "File.h"
#include "Filter.h"
#include "Geometry.h"
#include "HDR.h"
#include "Help.h"
#include "Image.h"
#include "LightField.h"
#include "Arithmetic.h"
#include "Network.h"
#include "NetworkOps.h"
#include "NeuralNetwork.h"
#include "Paint.h"
#include "Panorama.h"
#include "Parser.h"
#include "Prediction.h"
#include "Projection.h"
#include "Stack.h"
#include "Statistics.h"
#include "Wavelet.h"
#include "macros.h"
#include "tables.h"

}

#undef NO_SDL

#endif
