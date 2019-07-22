#ifndef PANORAMA_H
#define PANORAMA_H

class LoadPanorama : public Operation {
 public:
    void help();

    void parse(vector<string> args);

    static Image apply(string filename, 
                       float minTheta, float maxTheta,
                       float minPhi, float maxPhi,
                       int width, int height);
}; 

class PanoramaBackground : public Operation {
 public:
    void help();
    void parse(vector<string> args);
    static Image apply(Window im);
};

#endif
