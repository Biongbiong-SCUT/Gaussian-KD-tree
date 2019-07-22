#ifndef DEVICE_H
#define DEVICE_H

#ifdef NVCC
#define DEVICE_API
#else
#define DEVICE_API
#endif

typedef enum {INNER = 0, OUTER, ELEMENTWISE, MATRIX, SCALAR} convolve_t;

extern "C" {
    DEVICE_API void convolve3D(int dataFrames, int dataWidth, int dataHeight, int dataChannels, float *data, 
			       int kernelFrames, int kernelWidth, int kernelHeight, int kernelChannels, float *kernel,
			       float *output, convolve_t type = OUTER, int outputStride = 1);

    DEVICE_API void initCuda(int argc, char **argv);
}

#endif
