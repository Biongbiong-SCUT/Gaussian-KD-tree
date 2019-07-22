// tell cutil we want the CUDA_SAFE_CALL macro to actually do something
#define _DEBUG
#include "cutil.h"

#include "gpu_gkdtree.h"
#include <cuda_runtime.h>
#include <stdio.h>

#ifdef EMULATION
#define DEBUG(s) {s; fflush(stdout);}
#define DEBUG0(s) {if (threadIdx.x == 0) {s; fflush(stdout);}}
#else
#define DEBUG(s)
#define DEBUG0(s)
#endif

#define INF (1e99)

int *debugPtr;
__device__ int debug[1024*1024];

void initCuda(int argc, char **argv) {
    CUT_DEVICE_INIT(argc, argv);    

    cudaDeviceProp prop;
    CUDA_SAFE_CALL_NO_SYNC(cudaGetDeviceProperties(&prop, 0));
    printf("Device name: %s\n", prop.name);
    printf("Max threads per block: %i\n", prop.maxThreadsPerBlock);
    printf("Max threads dim: %i %i %i\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
    printf("Max grid size: %i %i %i \n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
    printf("Shared memory per block: %i Kb\n", prop.sharedMemPerBlock/1024);
    printf("Total global memory: %i Kb\n", prop.totalGlobalMem/1024);
    printf("Warp size: %i\n", prop.warpSize);
    printf("Memory pitch: %i\n", prop.memPitch);
    printf("Registers per block: %i\n", prop.regsPerBlock);
    printf("Clock rate: %i\n", prop.clockRate);
    printf("Texture alignment: %i\n", prop.textureAlignment);
    fflush(stdout);

    // allocate the debug array
    CUDA_SAFE_CALL(cudaGetSymbolAddress((void **)&debugPtr, debug));
    CUDA_SAFE_CALL(cudaMemset(debugPtr, 0, sizeof(int)*1024*1024));
}

struct node {
    int cut_dim;
    // the sign bits on left and right are used to indicate that the
    // child is another node (positive) or a leaf (negative).  If the
    // child is a node, the number is an index into the nodes array.
    // If it's a leaf, the number (stripped of the minus sign) is an
    // index into the data array.
    int left, right, parent;
    float max_val, cut_val, min_val, _pad;
};

struct kd_tree {
    node *nodeArray;
    float *leafPositions;
    float *leafValues;
    int nodeCount, leafCount;
    int positionDimensions, valueDimensions;
};

kd_tree *tree;
node *nodeArray;
float *leafPositions;
float *leafValues;
int maxLeaves;

#define BUILD_SIZE_LIMIT 0.15
#define MAX_DIMENSIONS 30
#define MAX_VALUE_DIMENSIONS 4
#define MAX_BUILD_BLOCKS (1<<15)

#include "gpu_gkdtree_util.cu"
#include "gpu_gkdtree_build.cu"
#include "gpu_gkdtree_include.cu"
#include "gpu_gkdtree_finalize.cu"
#include "gpu_gkdtree_gaussian_lookup.cu"
#include "gpu_gkdtree_splat.cu"
#include "gpu_gkdtree_slice.cu"
#include "gpu_gkdtree_blur.cu"



