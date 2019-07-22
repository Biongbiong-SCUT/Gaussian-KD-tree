#ifndef PAINT_H
#define PAINT_H

class Eval : public Operation {
  public:
    void help();
    void parse(vector<string> args);
    static Image apply(Window im, string expression);
};

class EvalChannels : public Operation {
  public:
    void help();
    void parse(vector<string> args);
    static Image apply(Window im, vector<string> expressions);
};

class Plot : public Operation {
  public:
    void help();
    void parse(vector<string> args);
    static Image apply(Window im, int width, int height, float lineThickness);
};

#endif
