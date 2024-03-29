#ifndef FILE_H
#define FILE_H

class Load : public Operation {
  public:
    void help();
    void parse(vector<string> args);
    static Image apply(string filename);
};

class LoadFrames : public Operation {
  public:
    void help();
    void parse(vector<string> args);
    static Image apply(vector<string> args);
};

class Save : public Operation {
  public:
    void help();
    void parse(vector<string> args);
    static void apply(Window im, string filename, string arg = "");
};

class SaveFrames : public Operation {
  public:
    void help();
    void parse(vector<string> args);
    static void apply(Window im, string pattern, string arg = "");
};

class LoadBlock : public Operation {
  public:
    void help();
    void parse(vector<string> args);
    static Image apply(string filename, int t, int x, int y, int c, 
		       int frames, int width, int height, int channels);
};

class SaveBlock : public Operation {
  public:
    void help();
    void parse(vector<string> args);
    static void apply(Window im, string filename, int t, int x, int y, int c);
};

class CreateTmp : public Operation {
  public:
    void help();
    void parse(vector<string> args);
    static void apply(string filename, int frames, int width, int height, int channels);
};

class LoadArray : public Operation {
  public:
    void help();
    void parse(vector<string> args);

    template<typename T> 
	static Image apply(string filename, int frames, int width, int height, int channels);
};

class SaveArray : public Operation {
  public:
    void help();
    void parse(vector<string> args);
    
    template<typename T>
	static void apply(Window im, string filename);
};

namespace FileEXR {
    void help();
    Image load(string filename);
    void save(Window im, string filename, string compression);
}

namespace FileHDR {
    void help();
    void save(Window im, string filename);
    Image load(string filename);
}

namespace FileJPG {
    void help();
    void save(Window im, string filename, int quality);
    Image load(string filename);
}

namespace FilePNG {
    void help();
    Image load(string filename);
    void save(Window im, string filename);
}

namespace FilePPM {
    void help();
    Image load(string filename);
    void save(Window im, string filename, int depth);
}

namespace FileRAW {
    void help();
    Image load(string filename);
}

namespace FileTIFF {
    void help();
    Image load(string filename);
    void save(Window im, string filename, string type);
}

namespace FileTGA {
    void help();
    Image load(string filename);
    void save(Window im, string filename);
}

namespace FileTMP {
    void help();
    void save(Window im, string filename);
    Image load(string filename);
}

namespace FileYUV {
    void help();
    void save(Window im, string filename);
    Image load(string filename);
}

namespace FileWAV {
    void help();
    void save(Window im, string filename);
    Image load(string filename);
}

#endif
