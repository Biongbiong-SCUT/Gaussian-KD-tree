#ifndef HELP_H
#define HELP_H

class Help : public Operation {
public:
    void help();
    void parse(vector<string> args);
};

#endif
