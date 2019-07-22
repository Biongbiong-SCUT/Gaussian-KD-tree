#ifndef CONVOLVE_H
#define CONVOLVE_H

class Convolve : public Operation {
  public:
    void help();
    void parse(vector<string> args);
    static Image apply(Window im, Window filter);
};

class Deconvolve : public Operation {
  public:
    void help();
    void parse(vector<string> args);
    static Image apply(Window im, Window filter, float time);
};

class DeconvolveCG : public Operation {
 public:
    void help();
    void parse(vector<string> args);
    static Image apply(Window im, Window filter, float time);
    static Image apply(Window im, Window filter, float time, Window initialGuess);
 private:
    static double dot(Image &a, Image &b);
    static double norm(Image &a);
    static void scaleAdd(Image &out, float alpha, Image &a, Image &b);
};

#endif
