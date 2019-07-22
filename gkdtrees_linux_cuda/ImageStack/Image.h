#ifndef IMAGE_H
#define IMAGE_H

#include "tables.h"

class Window {
  public:
    Window(Window im, int mint_, int minx_, int miny_, int frames_, int width_, int height_) {
	int mint = MAX(0, mint_);
	int maxt = MIN(im.frames, mint_ + frames_);
	int minx = MAX(0, minx_);
	int maxx = MIN(im.width, minx_ + width_);
	int miny = MAX(0, miny_);
	int maxy = MIN(im.height, miny_ + height_);

	xstride = im.xstride;
	ystride = im.ystride;
	tstride = im.tstride;

	width = maxx - minx;
	height = maxy - miny;
	frames = maxt - mint;
	channels = im.channels;

	data = im.data + mint * tstride + miny * ystride + minx * xstride;
    }

    float *operator()(int t, int x, int y) {
	return data + t * tstride + x * xstride + y * ystride;
    }

    float *operator()(int x, int y) {
	return data + x * xstride + y * ystride;
    }

    float *operator()(int x) {
	return data + x * xstride;
    }

    void sample2D(int t, float fx, float fy, float *result) {
	fx -= 0.5;
	fy -= 0.5;
	int ix = (int)fx;
	int iy = (int)fy;
	const int LEFT = -2;
	const int RIGHT = 3;
	const int WIDTH = 6;
	int minX = ix + LEFT;
	int maxX = ix + RIGHT;
	int minY = iy + LEFT;
	int maxY = iy + RIGHT;

	float weightX[WIDTH];
	float weightY[WIDTH];
	float totalXWeight = 0, totalYWeight = 0;
	for (int x = 0; x < WIDTH; x++) {
	    float diff = (fx - (x + ix + LEFT)); // ranges between +/- RIGHT
	    float val = lanczos_3(diff);
	    weightX[x] = val;
	    totalXWeight += val;
	}

	for (int y = 0; y < WIDTH; y++) {
	    float diff = (fy - (y + iy + LEFT)); // ranges between +/- RIGHT
	    float val = lanczos_3(diff);
	    weightY[y] = val;
	    totalYWeight += val;
	}

	totalXWeight = 1.0f/totalXWeight;
	totalYWeight = 1.0f/totalYWeight;

	for (int i = 0; i < WIDTH; i++) {
	    weightX[i] *= totalXWeight;
	    weightY[i] *= totalYWeight;
	}

	for (int c = 0; c < channels; c++) {
	    result[c] = 0;
	}


	float *yWeightPtr = weightY;
	for (int y = minY; y <= maxY; y++) {
	    float *xWeightPtr = weightX;
	    int sampleY = MIN(MAX(0, y), height-1);
	    for (int x = minX; x <= maxX; x++) {
		int sampleX = MIN(MAX(0, x), width-1);
		float yxWeight = (*yWeightPtr) * (*xWeightPtr);
		float *ptr = (*this)(t, sampleX, sampleY);
		for (int c = 0; c < channels; c++) {
		    result[c] += ptr[c] * yxWeight;
		}
		xWeightPtr++;
	    }
	    yWeightPtr++;
	}      	
    }

    void sample2D(float fx, float fy, float *result) {
	sample2D(0, fx, fy, result);
    }


    void sample2DLinear(float fx, float fy, float *result) {
	sample2DLinear(0, fx, fy, result);
    }

    void sample2DLinear(int t, float fx, float fy, float *result) {
	fx -= 0.5;
	fy -= 0.5;
	int ix = (int)fx;
	int iy = (int)fy;
	fx -= ix;
	fy -= iy;

	float *ptr = data + t * tstride + iy * ystride + ix * xstride;
	for (int c = 0; c < channels; c++) {
	    float s1 = (1-fx) * ptr[c] + fx * ptr[c + xstride];
	    float s2 = (1-fx) * ptr[c + ystride] + fx * ptr[c + xstride + ystride];
	    result[c] = (1-fy) * s1 + fy * s2;
	    
	}

    }

    void sample3DLinear(float ft, float fx, float fy, float *result) {
	fx -= 0.5;
	fy -= 0.5;
	ft -= 0.5;
	int ix = (int)fx;
	int iy = (int)fy;
	int it = (int)ft;
	fx -= ix;
	fy -= iy;
	ft -= it;

	float *ptr = data + it * tstride + iy * ystride + ix * xstride;
	for (int c = 0; c < channels; c++) {
	    float s11 = (1-fx) * ptr[c] + fx * ptr[c + xstride];
	    float s12 = (1-fx) * ptr[c + ystride] + fx * ptr[c + xstride + ystride];
	    float s1 = (1-fy) * s11 + fy * s12;

	    float s21 = (1-fx) * ptr[c + tstride] + fx * ptr[c + xstride + tstride];
	    float s22 = (1-fx) * ptr[c + ystride + tstride] + fx * ptr[c + xstride + ystride + tstride];
	    float s2 = (1-fy) * s21 + fy * s22;

	    result[c] = (1-ft) * s1 + ft * s2;	   
	}
	
    }

    void sample3D(float ft, float fx, float fy, float *result) {
	fx -= 0.5;
	fy -= 0.5;
	ft -= 0.5;
	int ix = (int)fx;
	int iy = (int)fy;
	int it = (int)ft;
	const int LEFT = -2;
	const int RIGHT = 3;
	const int WIDTH = 6;
	int minX = ix + LEFT;
	int maxX = ix + RIGHT;
	int minY = iy + LEFT;
	int maxY = iy + RIGHT;
	int minT = it + LEFT;
	int maxT = it + RIGHT;
	float weightX[WIDTH];
	float weightY[WIDTH];
	float weightT[WIDTH];

	float totalXWeight = 0, totalYWeight = 0, totalTWeight = 0;

	for (int x = 0; x < WIDTH; x++) {
	    float diff = (fx - (x + ix + LEFT)); // ranges between +/- RIGHT
	    float val = lanczos_3(diff);
	    weightX[x] = val;
	    totalXWeight += val;
	}

	for (int y = 0; y < WIDTH; y++) {
	    float diff = (fy - (y + iy + LEFT)); // ranges between +/- RIGHT
	    float val = lanczos_3(diff);
	    weightY[y] = val;
	    totalYWeight += val;
	}

	for (int t = 0; t < WIDTH; t++) {
	    float diff = (ft - (t + it + LEFT)); // ranges between +/- RIGHT
	    float val = lanczos_3(diff);
	    weightT[t] = val;
	    totalTWeight += val;
	}

	totalXWeight = 1.0f/totalXWeight;
	totalYWeight = 1.0f/totalYWeight;
	totalTWeight = 1.0f/totalTWeight;

	for (int i = 0; i < WIDTH; i++) {
	    weightX[i] *= totalXWeight;
	    weightY[i] *= totalYWeight;
	    weightT[i] *= totalTWeight;
	}

	for (int c = 0; c < channels; c++) {
	    result[c] = 0;
	}

	float *tWeightPtr = weightT;
	for (int t = minT; t <= maxT; t++) {
	    int sampleT = MIN(MAX(t, 0), frames-1);
	    float *yWeightPtr = weightY;
	    for (int y = minY; y <= maxY; y++) {
		int sampleY = MIN(MAX(y, 0), height-1);
		float tyWeight = (*yWeightPtr) * (*tWeightPtr);
		float *xWeightPtr = weightX;
		for (int x = minX; x <= maxX; x++) {
		    int sampleX = MIN(MAX(x, 0), width-1);
		    float tyxWeight = tyWeight * (*xWeightPtr);
		    float *ptr = (*this)(sampleT, sampleX, sampleY);
		    for (int c = 0; c < channels; c++) {
			result[c] += ptr[c] * tyxWeight;
		    }
		    xWeightPtr++;
		}
		yWeightPtr++;
	    }
	    tWeightPtr++;
	}    

    }

    int width, height, frames, channels;
    int xstride, ystride, tstride;
    float *data;    

  protected:
    Window() {}
};

class Image : public Window {
  public:
    Image() : refCount(NULL) {
	width = frames = height = channels = 0;
	xstride = ystride = tstride = 0;
	data = NULL;
    }
    
	/*
    void debug() {
        printf("%llx(%i@%llx): %i %i %i %i\n", (unsigned long long)data, refCount[0], (unsigned long long)refCount, frames, width, height, channels);
    }
	*/    

    Image(int frames_, int width_, int height_, int channels_, const float *data_ = NULL) {
	frames = frames_;
	width = width_;
	height = height_;
	channels = channels_;

	long long memory = ((long long)frames_ * 
			    (long long)height_ *
			    (long long)width_ * 
			    (long long)channels_);

        data = new float[memory];
        if (!data_) memset(data, 0, memory * sizeof(float));
        else memcpy(data, data_, memory * sizeof(float));
        
        xstride = channels;
        ystride = xstride * width;
        tstride = ystride * height;
        refCount = new int;
        *refCount = 1;
        
        //printf("Making new image "); 
        //debug();
    }
    
    // does not copy data

    Image &operator=(const Image &im) {
	if (refCount) {
	    refCount[0]--;
	    if (*refCount <= 0) {
		delete refCount;
		delete[] data;
	    }
	}

        width = im.width;
        height = im.height;
        channels = im.channels;
        frames = im.frames;
        
        data = im.data;
        
        xstride = channels;
        ystride = xstride * width;
        tstride = ystride * height;	
        
        refCount = im.refCount;       
        if (refCount) refCount[0]++;
        
        return *this;
    }
    
    Image(const Image &im) {
        width = im.width;
        height = im.height;
        channels = im.channels;
        frames = im.frames;
        
        data = im.data;       
        xstride = channels;
        ystride = xstride * width;
        tstride = ystride * height;	
        
        refCount = im.refCount;        
        if (refCount) refCount[0]++;       
    }
    
    // copies data from the window
    Image(Window im) {
        width = im.width;
        height = im.height;
        channels = im.channels;
        frames = im.frames;
        
        xstride = channels;
        ystride = xstride * width;
        tstride = ystride * height;	
        
        refCount = new int;
        *refCount = 1;
	long long memory = ((long long)width *
			    (long long)height *
			    (long long)channels *
			    (long long)frames);
        data = new float[memory];
        
        for (int t = 0; t < frames; t++) {
            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++) {
                    for (int c = 0; c < channels; c++) {
                        (*this)(t, x, y)[c] = im(t, x, y)[c];
                    }
                }
            }
        }
        
    }
    
    // makes a new copy of this image
    Image copy() {
        return Image(*((Window *)this));
    }
    
    ~Image();

    int *refCount;
    
  protected:
    Image &operator=(Window im) {
        return *this;
    }
};

#endif
