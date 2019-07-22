#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#include "Dual.h"

using namespace std;

class NeuralNetwork {
    vector< vector<Dual> > weight_;
    vector<int> layerSize_;

    void initialize(vector<int> layerSizes);

  public:

    NeuralNetwork(int inputNodes_, int hiddenNodes_, int outputNodes_) {
	vector<int> sizes(3);
	sizes[0] = inputNodes_;
	sizes[1] = hiddenNodes_;
	sizes[2] = outputNodes_;
	initialize(sizes);
    }

    NeuralNetwork(int inputNodes_, int hiddenNodes1_, int hiddenNodes2_, int outputNodes_) {
	vector<int> sizes(4);
	sizes[0] = inputNodes_;
	sizes[1] = hiddenNodes1_;
	sizes[2] = hiddenNodes2_;
	sizes[3] = outputNodes_;
	initialize(sizes);
    }

    NeuralNetwork(vector<int> layerSizes) {
	initialize(layerSizes);
    }

    void apply(vector<float> input, vector<float> &output);
    Dual error(vector<float> input, vector<float> output);
    float train(vector<float> input, vector<float> output, float stepSize);    
    void backPropogate(Dual error, float stepSize);

    int inputNodes() {return layerSize_[0];}
    int outputNodes() {return layerSize_[layerSize_.size()-1];}
    int layerSize(int layer) {return layerSize_[layer];}

    void removeNode(int layer);
    void addNode(int layer);    

    float &weight(int layer, int inputNode, int outputNode);
};


#endif

