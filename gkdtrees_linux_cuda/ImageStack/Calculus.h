#ifndef CALCULUS_H
#define CALCULUS_H

class Gradient : public Operation {
  public:
    void help();
    void parse(vector<string> args);
    static void apply(Window im, string dimensions);
    static void apply(Window im, char dimension);
};

class Integrate : public Operation {
  public:
    void help();
    void parse(vector<string> args);
    static void apply(Window im, string dimensions);
    static void apply(Window im, char dimension);
};

class Laplacian : public Operation {
  public:
    void help();
    void parse(vector<string> args);
    static Image apply(Window im);
};

class Poisson : public Operation {
  public:
    void help();
    void parse(vector<string> args);
    static Image apply(Window dx, Window dy, float termination = 0.01);
    static Image apply(Window dt, Window dx, Window dy, float termination = 0.01);
};

#endif
