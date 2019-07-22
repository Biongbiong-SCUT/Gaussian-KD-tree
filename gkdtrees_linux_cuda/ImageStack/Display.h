#ifndef DISPLAY_H
#define DISPLAY_H

class Display : public Operation {
  public:
    ~Display();
    void help();
    void parse(vector<string> args);
    static void apply(Window im, bool fullscreen = false);
};



#endif
