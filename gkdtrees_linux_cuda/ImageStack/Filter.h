#ifndef FILTER_H
#define FILTER_H

class GaussianBlur : public Operation {
  public:
    void help();
    void parse(vector<string> args);
    static Image apply(Window im, float filterFrames, float filterWidth, float filterHeight);
};

class ChromaBlur : public Operation {
  public:
    void help();
    void parse(vector<string> args);
    static Image apply(Window im, float spatialSigma, float colorSigma);
};

class LanczosBlur : public Operation {
  public:
    void help();
    void parse(vector<string> args);
    static Image apply(Window im, float filterFrames, float filterWidth, float filterHeight);
};

class FastBlur : public Operation {
  public:
    void help();
    void parse(vector<string> args);
    static void apply(Window im, float filterFrames, float filterWidth, float filterHeight);
  protected: // these functions are used by subclasses that do rect filters
    static void blurXCompletely(Window im);
    static void blurX(Window im, int width, int iterations);
    static void blurY(Window im, int width, int iterations);
    static void blurT(Window im, int width, int iterations);
};

class RectFilter : public FastBlur {
  public:
    void help();
    void parse(vector<string> args);
    static void apply(Window im, int filterFrames, int filterWidth, int filterHeight);
};

class FastBandlimit : public FastBlur {
  public:
    void help();
    void parse(vector<string> args);
    static Image apply(Window im, float filterFrames, float filterWidth, float filterHeight);
 private:
    static Image applyOnce(Window im, float filterFrames, float filterWidth, float filterHeight);
};

class Bilateral : public Operation {
  public:
    void help();
    void parse(vector<string> args);
    static Image apply(Window im, float spatialSigma, float colorSigma);
};

class JointBilateral : public Operation {
  public:
    void help();
    void parse(vector<string> args);
    static Image apply(Window im1, Window reference, float spatialSigma, float colorSigma);
};

class BilateralGrid : public Operation {
  public:
    void help();
    void parse(vector<string> args);
    static Image apply(Window im, int xBuckets, int yBuckets, int cBuckets);
};


class JointBilateralGrid : public Operation {
  public:
    void help();
    void parse(vector<string> args);
    static Image apply(Window im, Window reference, int xBuckets, int yBuckets, int cBuckets);    
};


class SliceBilateralGrid : public Operation {
  public:
    void help();
    void parse(vector<string> args);
    static Image apply(Window grid, Window reference);    
};

class BilateralSharpen : public Operation {
  public:
    void help();
    void parse(vector<string> args);
    static Image apply(Window im, float spatialSigma, float colorSigma, float sharpness);
};

class MedianFilter : public Operation {
  public:
    void help();
    void parse(vector<string> args);
    static Image apply(Window im, int radius);
};

class PercentileFilter : public Operation {
  public:
    void help();
    void parse(vector<string> args);
    static Image apply(Window im, int radius, float percentile);
};

class CircularFilter : public Operation {
  public:
    void help();
    void parse(vector<string> args);
    static Image apply(Window im, int radius);
};

#endif
