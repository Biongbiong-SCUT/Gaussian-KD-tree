#ifndef GPU_GKDTREE_H
#define GPU_GKDTREE_H

#ifdef NVCC
#else
#endif

#define DEVICE_API

extern "C" {
    DEVICE_API void initCuda(int argc, char **argv);

    DEVICE_API void buildTree(int posDimensions, int valDimensions, float *data, int nData, int nAllocated);

    DEVICE_API void include(float *positions, int nPositions);

    DEVICE_API void finalizeTree();
    DEVICE_API void destroyTree();

    DEVICE_API void gaussianSplat(float *positions, float *values, int nPositions, 
				  int samples, float sigma);


    DEVICE_API void gaussianBlur(int samples, float sigma);

    DEVICE_API void gaussianSlice(float *positions, float *values, int nPositions, 
				  int samples, float sigma);


    //DEVICE_API void growTree(float *data, int dimensions, int nData);
    //DEVICE_API int getLeaves();

}

#endif
