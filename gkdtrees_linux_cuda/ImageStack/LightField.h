#ifndef LIGHTFIELD_H
#define LIGHTFIELD_H

// a LightField is a window which assumes u and v are rolled up into x and y, like an image of the lenslets
class LightField : public Window {
public:
    LightField(Window im, int uSize_, int vSize_) : Window(im), uSize(uSize_), vSize(vSize_) {
	assert(width % uSize == 0, "width is not a multiple of lenslet width\n");
	assert(height % vSize == 0, "height is not a multiple of lenslet height\n");	       
	xSize = width / uSize;
	ySize = height / vSize;
    }

    float *operator()(int t, int x, int y, int u, int v) {
	return data + t * tstride + (x*uSize + u) * xstride + (y*vSize + v) * ystride;
    }

    float *operator()(int x, int y, int u, int v) {
	return data + (x*uSize + u) * xstride + (y*vSize + v) * ystride;
    }

    // quadrilinear 4D sampling (quadriLanczos3 too expensive, 6^4=1296)
    // x,y,u,v follow the same coordinate conventions as
    // operator()
    void sample4D(int t, float x, float y, float u, float v, float *result) {
      int ix[2], iy[2], iu[2], iv[2]; // integer indices
      float wx[2], wy[2], wu[2], wv[2]; // weighting factors

      if( (x < -0.5 || y < -0.5 || x > xSize-0.5 || y > ySize-0.5)
          || (u < -0.5 || v < -0.5 || u > uSize-0.5 || v > vSize-0.5) ) {
        // out of bounds, so return zero
        for(int c=0;c<channels;c++) {
          result[c]=0;
        }
        return;
      }
        
      ix[0] = (int)(floor(x));
      iy[0] = (int)(floor(y));
      iu[0] = (int)(floor(u));
      iv[0] = (int)(floor(v));
      // clamp against bounds
      ix[0] = MIN(MAX(ix[0],0),xSize-1);
      iy[0] = MIN(MAX(iy[0],0),ySize-1);
      iu[0] = MIN(MAX(iu[0],0),uSize-1);
      iv[0] = MIN(MAX(iv[0],0),vSize-1);

      ix[1] = ix[0]+1;
      iy[1] = iy[0]+1;
      iu[1] = iu[0]+1;
      iv[1] = iv[0]+1;
      // clamp against bounds
      ix[1] = MIN(ix[1],xSize-1);
      iy[1] = MIN(iy[1],ySize-1);
      iu[1] = MIN(iu[1],uSize-1);
      iv[1] = MIN(iv[1],vSize-1);

      // calculate the weights for quadrilinear
      wx[1] = x-ix[0];
      wy[1] = y-iy[0];
      wu[1] = u-iu[0];
      wv[1] = v-iv[0];
      wx[0] = 1-wx[1];
      wy[0] = 1-wy[1];
      wu[0] = 1-wu[1];
      wv[0] = 1-wv[1];
      
      // do the computation
      for(int c=0;c<channels;c++) {
        result[c]=0;
      }

      for(int i=0;i<2;i++) { // go through iu
        for(int j=0;j<2;j++) { // go through ix
          for(int k=0;k<2;k++) { // go through iv
            for(int l=0;l<2;l++) { // go through iy
              for(int c=0;c<channels;c++) {
                result[c] += (*this)(t,ix[j],iy[l],iu[i],iv[k])[c] * 
                  wx[j]*wy[l]*wu[i]*wv[k];
              }
            }
          }
        }
      }

      /*
      int ix, iy, iu, iv; // floored versions of the input arguments
      float wx, wy, wu, wv; // weighting for those floored coordinates
      ix = (int)(floor(x));
      iy = (int)(floor(y));
      iu = (int)(floor(u));
      iv = (int)(floor(v));
      wx = 1-x+ix;
      wy = 1-y+iy;
      wu = 1-u+iu;
      wv = 1-v+iv;
      for(int c=0;c<channels;c++) {
        result[c]=0;
        result[c] += (*this)(t,ix,iy,iu,iv)[c] * (wx)*(wy)*(wu)*(wv);
        result[c] += (*this)(t,ix,iy,iu+1,iv)[c] * (wx)*(wy)*(1-wu)*(wv);
        result[c] += (*this)(t,ix+1,iy,iu,iv)[c] * (1-wx)*(wy)*(wu)*(wv);
        result[c] += (*this)(t,ix+1,iy,iu+1,iv)[c] * (1-wx)*(wy)*(1-wu)*(wv);
        result[c] += (*this)(t,ix,iy,iu,iv+1)[c] * (wx)*(wy)*(wu)*(1-wv);
        result[c] += (*this)(t,ix,iy,iu+1,iv+1)[c] * (wx)*(wy)*(1-wu)*(1-wv);
        result[c] += (*this)(t,ix+1,iy,iu,iv+1)[c] * (1-wx)*(wy)*(wu)*(1-wv);
        result[c] += (*this)(t,ix+1,iy,iu+1,iv+1)[c] * (1-wx)*(wy)*(1-wu)*(1-wv);
        result[c] += (*this)(t,ix,iy+1,iu,iv)[c] * (wx)*(1-wy)*(wu)*(wv);
        result[c] += (*this)(t,ix,iy+1,iu+1,iv)[c] * (wx)*(1-wy)*(1-wu)*(wv);
        result[c] += (*this)(t,ix+1,iy+1,iu,iv)[c] * (1-wx)*(1-wy)*(wu)*(wv);
        result[c] += (*this)(t,ix+1,iy+1,iu+1,iv)[c] * (1-wx)*(1-wy)*(1-wu)*(wv);
        result[c] += (*this)(t,ix,iy+1,iu,iv+1)[c] * (wx)*(1-wy)*(wu)*(1-wv);
        result[c] += (*this)(t,ix,iy+1,iu+1,iv+1)[c] * (wx)*(1-wy)*(1-wu)*(1-wv);
        result[c] += (*this)(t,ix+1,iy+1,iu,iv+1)[c] * (1-wx)*(1-wy)*(wu)*(1-wv);
        result[c] += (*this)(t,ix+1,iy+1,iu+1,iv+1)[c] * (1-wx)*(1-wy)*(1-wu)*(1-wv);
      }
      */
    }

  void sample4D(float x, float y, float u, float v, float *result) {
    sample4D(0,x,y,u,v,result);
  }
    
    int uSize, vSize;
    int xSize, ySize;
};

class LFRectify : public Operation {
public:
    void help();
    void parse(vector<string> args);

    // Rectify a light field from scratch
    static Image apply(Window im, int estimateLensletSize, int desiredLensletSize); 

    // Rectify a light field using a saved parameter file
    static Image apply(Window im, string filename); 

    // Compute and save the appropriate warp to rectify a light field
    static void apply(Window im, int estimateLensletSize, int desiredLensletSize, string filename);
private:
    class Solution;
};

class LFRectify2 : public Operation {
public:
    void help();
    void parse(vector<string> args);

    // Rectify a light field from scratch
    static Image apply(Window im, int estimateLensletSize, int desiredLensletSize, bool nearestNeighbour = false);

    // Rectify a light field using a saved parameter file
    static Image apply(Window im, string filename, bool nearestNeighbour = false); 

    // Compute and save the appropriate warp to rectify a light field
    static void apply(Window im, int estimateLensletSize, int desiredLensletSize, string filename);

    class LensletCenter {
      public:
        LensletCenter() :
            x(0), y(0), lx(0), ly(0), 
            left(NULL), right(NULL), up(NULL), down(NULL),          
            touched(false) {};
        LensletCenter(int ix, int iy) : 
            x(ix), y(iy), lx(0), ly(0), 
            left(NULL), right(NULL), up(NULL), down(NULL),          
            touched(false) {};
            
        // image coordinates
        int x, y;

        // lenslet coordinates and neighbour info, initially
        // undefined, will be filled in by the compute method of CubicWarp
        int lx, ly; 
        LensletCenter *left, *right, *up, *down; 

        // a flag to help with graph traversal
        bool touched;
    };

    class CubicWarp {
      public:
        // the model has 20 DOF
        // x' = a + cx + ey + gx^2 + ixy + ky^2 + mx^3 + ox^2y + qxy^2 + sy^3
        // y' = b + dx + fy + hx^2 + jxy + ly^2 + nx^3 + px^2y + rxy^2 + ty^3
        
        // this maps lenslet coords to image coords according to the above equation
        // note that it's column major
        double model[20];    
        
        // this maps image coords to lenslet coords in the same way
        double invmodel[20];       

        // This extracts lenslet centers from an image and computes a warp
        void compute(Window im, float estimatedLensletSize, int desiredLensletSize);

        // This computes a warp using given lenslet centers
        void compute(LensletCenter *centers, int centerCount, float estimatedLensletSize, int desiredLensletSize);

        // Warp an image of a lenslet array using the computed inverse model.
        Image apply(Window im, bool nearestNeighbour = false);

        // functions to map back and forth between lenslet and image coordinates
        void lensletToImage(float lx, float ly, float *ix, float *iy);
        void imageToLenslet(float ix, float iy, float *lx, float *ly);

        // load or save the warp from or to a file
        void load(string filename);
        void save(string filename);

        // print the model or inverse model to stdout
        void printModel() {print(model);}
        void printInverseModel() {print(invmodel);}

        void print(double *m) {
	    printf("X' = %f + %fx + %fy + \n"
		   "     %fx^2 + %fxy + %fy^2 + \n"
		   "     %fx^3 + %fx^2y + %fxy^2 + %fy^3\n"
		   "\n"
		   "Y' = %f + %fx + %fy + \n"
		   "     %fx^2 + %fxy + %fy^2 + \n"
		   "     %fx^3 + %fx^2y + %fxy^2 + %fy^3\n"
		   "\n",
		   m[0], m[2], m[4], m[6], m[8], 
		   m[10], m[12], m[14], m[16], m[18],
		   m[1], m[3], m[5], m[7], m[9], 
		   m[11], m[13], m[15], m[17], m[19]);
        }

        // The desired lenslet size.
        // Even though this isn't part of the warp per se, it's
        // convenient to save and load it in the same file, and
        // remember it for the image warp operation
        int desiredLensletSize;
    };

};

class LFFocalStack : public Operation {
public:
    void help();
    void parse(vector<string> args);
    static Image apply(LightField im, float minAlpha, float maxAlpha, float deltaAlpha);
};

class LFFocus : public Operation {
public:
    void help();
    void parse(vector<string> args);
    static Image apply(LightField im, float alpha, float beta = 1);
};

class LFSuperResolution : public Operation {
public:
    void help();
    void parse(vector<string> args);
    static Image apply(LightField lf, int factor, int minDepth = 0, int maxDepth = 0);
};

class LFWarp : public Operation {
  public:
    void help();
    void parse(vector<string> args);
    static Image apply(LightField lf, Window warper, bool quick);
};

class LFGeom : public Operation {
  public:
    void help();
    void parse(vector<string> args);
    static Image applyGeneric(int width, int height, double *A, double *B, double *C, double *D, double *a0, double *b0, double *c0, double *d0);
};

class LFProject : public Operation {
  public:
    void help();
    void parse(vector<string> args);
    static Image apply(LightField lf, float *matrix);
};

class LFPoint : public Operation {
  public:
    void help();
    void parse(vector<string> args);
    static void apply(LightField lf, float x, float y, float z);
};

class LFFocalStack2 : public Operation {
public:
    void help();
    void parse(vector<string> args);
    static Image apply(LightField im, float minAlpha, float maxAlpha, float deltaAlpha, float apertureSize);
};
#endif
