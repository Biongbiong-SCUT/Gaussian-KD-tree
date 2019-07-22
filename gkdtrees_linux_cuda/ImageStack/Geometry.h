#ifndef GEOMETRY_H
#define GEOMETRY_H

class Upsample : public Operation {
  public:
    void help();
    void parse(vector<string> args);
    static Image apply(Window im, int boxWidth, int boxHeight);
    static Image apply(Window im, int boxFrames, int boxWidth, int boxHeight);
};

class Downsample : public Operation {
  public:
    void help();
    void parse(vector<string> args);
    static Image apply(Window im, int boxWidth, int boxHeight);
    static Image apply(Window im, int boxFrames, int boxWidth, int boxHeight);
};

class Subsample : public Operation {
  public:
    void help();
    void parse(vector<string> args);
    static Image apply(Window im, int boxFrames, int boxWidth, int boxHeight, 
		       int offsetT, int offsetX, int offsetY);
    static Image apply(Window im, int boxWidth, int boxHeight, 
		       int offsetX, int offsetY);
    static Image apply(Window im, int boxFrames, int offsetT);
};

class Riffle : public Operation {
  public:
    void help();
    void parse(vector<string> args);
    static void apply(Window im, int rt);
    static void apply(Window im, int rx, int ry);
    static void apply(Window im, int rt, int rx, int ry);
};

class Unriffle : public Operation {
  public:
    void help();
    void parse(vector<string> args);
    static void apply(Window im, int it);
    static void apply(Window im, int ix, int iy);
    static void apply(Window im, int it, int ix, int iy);
};

class Resample : public Operation {
  public:
    void help();
    void parse(vector<string> args);
    static Image apply(Window im, int width, int height);
    static Image apply(Window im, int frames, int width, int height);
 private:
    static Image resampleT(Window im, int frames);
    static Image resampleX(Window im, int width);
    static Image resampleY(Window im, int height);    
};

class Rotate : public Operation {
  public:
    void help();
    void parse(vector<string> args);
    static Image apply(Window im, float degrees);
};

class AffineWarp : public Operation {
  public:
    void help();
    void parse(vector<string> args);
    static Image apply(Window im, vector<double> warp);
};

class Crop : public Operation {
  public:
    void help();
    void parse(vector<string> args);
    static Image apply(Window im, int minX, int minY, int width, int height);
    static Image apply(Window im, int minFrame, int minX, int minY, int frames, int width, int height);
    static Image apply(Window im);
};

class Flip : public Operation {
  public:
    void help();
    void parse(vector<string> args);
    static void apply(Window im, char dimension);
};

class Adjoin : public Operation {
  public:
    void help();
    void parse(vector<string> args);
    static Image apply(Window a, Window b, char dimension);
};

class Transpose : public Operation {
  public:
    void help();
    void parse(vector<string> args);
    static Image apply(Window im, char arg1, char arg2);
};

class Translate : public Operation {
  public:
    void help();
    void parse(vector<string> args);
    static Image apply(Window im, int xoff, int yoff);
    static Image apply(Window im, int toff, int xoff, int yoff);
};

class Paste : public Operation {
  public:
    void help();
    void parse(vector<string> args);
    static void apply(Window into, Window from, int xdst, int ydst, 
		      int xsrc, int ysrc, int width, int height);
    static void apply(Window into, Window from, int tdst, int xdst, int ydst);
    static void apply(Window into, Window from, int xdst, int ydst);
    static void apply(Window into, Window from, int tdst, int xdst, int ydst,
		      int tsrc, int xsrc, int ysrc, int frames, int width, int height);
};

class Expand : public Operation {
  public:
    void help();
    void parse(vector<string> args);
    static Image apply(Window im, int minX, int minY, int width, int height);
    static Image apply(Window im, int minT, int minX, int minY, int frames, int width, int height);
};

class Tile : public Operation {
  public:
    void help();
    void parse(vector<string> args);
    static Image apply(Window im, int xTiles, int yTiles);
    static Image apply(Window im, int tTiles, int xTiles, int yTiles);
};

class TileFrames : public Operation {
  public:
    void help();
    void parse(vector<string> args);
    static Image apply(Window im, int xTiles, int yTiles);
};

class FrameTiles : public Operation {
  public:
    void help();
    void parse(vector<string> args);
    static Image apply(Window im, int xTiles, int yTiles);
};

class Warp : public Operation {
  public:
    void help();
    void parse(vector<string> args);
    static Image apply(Window coords, Window source);
};

class Reshape : public Operation {
  public:
    void help();
    void parse(vector<string> args);
    static void apply(Window im, int newFrames, int newWidth, int newHeight, int newChannels);
    static void apply(Image &im, int newFrames, int newWidth, int newHeight, int newChannels);
};

#endif
