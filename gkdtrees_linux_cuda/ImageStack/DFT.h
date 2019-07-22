#ifndef NO_FFTW
#ifndef DFT_H
#define DFT_H

class DCT : public Operation {
  public:
    void help();
    void parse(vector<string> args);
    static Image apply(Window win);
};

#endif
#endif
