#ifndef MATH_H
#define MATH_H

class Add : public Operation {
  public:
    void help();
    void parse(vector<string> args);
    static void apply(Window a, Window b);
};

class Multiply : public Operation {
  public:
    void help();
    void parse(vector<string> args);
    static void apply(Window a, Window b);
};

class Subtract : public Operation {
  public:
    void help();
    void parse(vector<string> args);
    static void apply(Window a, Window b);
};

class Divide : public Operation {
  public:
    void help();
    void parse(vector<string> args);
    static void apply(Window a, Window b);
};

class Maximum : public Operation {
  public:
    void help();
    void parse(vector<string> args);
    static void apply(Window a, Window b);
};

class Minimum : public Operation {
  public:
    void help();
    void parse(vector<string> args);
    static void apply(Window a, Window b);
};

class Log : public Operation {
  public:
    void help();
    void parse(vector<string> args);
    static void apply(Window a);
};

class Exp : public Operation {
  public:
    void help();
    void parse(vector<string> args);
    static void apply(Window a, float base = E);
};

class Abs : public Operation {
  public:
    void help();
    void parse(vector<string> args);
    static void apply(Window a);
};

class Offset : public Operation {
  public:
    void help();
    void parse(vector<string> args);
    static void apply(Window a, vector<float>);
    static void apply(Window a, float);
};

class Scale : public Operation {
  public:
    void help();
    void parse(vector<string> args);
    static void apply(Window a, vector<float>);
    static void apply(Window a, float);
};

class Gamma : public Operation {
  public:
    void help();
    void parse(vector<string> args);
    static void apply(Window a, vector<float>);
    static void apply(Window a, float);
};

class Mod : public Operation {
  public:
    void help();
    void parse(vector<string> args);
    static void apply(Window a, vector<float>);
    static void apply(Window a, float);
};

class Clamp : public Operation {
  public:
    void help();
    void parse(vector<string> args);
    static void apply(Window a, float lower = 0, float upper = 1);
};

class DeNaN : public Operation {
  public:
    void help();
    void parse(vector<string> args);
    static void apply(Window a, float replacement = 0);
};

class Threshold : public Operation {
  public:
    void help();
    void parse(vector<string> args);
    static void apply(Window a, float val);
};

class Normalize : public Operation {
  public:
    void help();
    void parse(vector<string> args);
    static void apply(Window a);
};

class Quantize : public Operation {
  public:
    void help();
    void parse(vector<string> args);
    static void apply(Window a, float increment);
};

#endif
