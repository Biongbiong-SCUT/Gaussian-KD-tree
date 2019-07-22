#ifndef PROJECTION_H
#define PROJECTION_H

class Sinugram : public Operation {
  public:
    void help();
    void parse(vector<string> args);
    static Image apply(Window im, int directions);
};

#endif
