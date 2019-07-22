#ifndef OPERATION_H
#define OPERATION_H

class Operation {
  public:
    virtual ~Operation() {};
    virtual void parse(vector<string>) = 0;
    virtual void help() = 0;
};

void loadOperations();
void unloadOperations();

#endif
