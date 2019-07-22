#ifndef CONTROL_H
#define CONTROL_H


class Loop : public Operation {
  public:
    void help();
    void parse(vector<string> args);
};

class Pause : public Operation {
  public:
    void help();
    void parse(vector<string> args);
};

#endif
