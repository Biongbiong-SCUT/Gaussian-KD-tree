#ifndef STACK_H
#define STACK_H

// these operations apply only to the stack, so they have no apply method

class Pop : public Operation {
  public:
    void help();
    void parse(vector<string> args);
};

class Push : public Operation {
  public:
    void help();
    void parse(vector<string> args);
};

class Pull : public Operation {
  public:
    void help();
    void parse(vector<string> args);
};

class Dup : public Operation {
  public:
    void help();
    void parse(vector<string> args);
};

#endif
