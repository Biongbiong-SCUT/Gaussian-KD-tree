#ifndef PREDICTION_H
#define PREDICTION_H

class Inpaint : public Operation {
  public:
    void help();
    void parse(vector<string> args);
    static Image apply(Window im, Window mask);
};

class Synthesize : public Operation {
  public:
    void help();
    void parse(vector<string> args);
    static Image apply(Window im, Window example, int iterations = 1);
};

#endif
