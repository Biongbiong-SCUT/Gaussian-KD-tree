// tell cutil we want the CUDA_SAFE_CALL macro to actually do something
#define _DEBUG
#include "cutil.h"

#include "device.h"
#include <cuda_runtime.h>
#include <stdio.h>

#define BLOCK_WIDTH 32
#define BLOCK_HEIGHT 1

texture<float, 3, cudaReadModeElementType> texKernel;
texture<float, 3, cudaReadModeElementType> texData;

template<convolve_t type>
static __global__ void _convolve3D(int dataWidth, int dataHeight, int dataChannels, int horizBlocks,
				   int kernelFrames, int kernelWidth, int kernelHeight, int kernelChannels, 
				   float *output, int outputStride) {

    // figure out what pixel I'm responsible for
    const int x = threadIdx.x + BLOCK_WIDTH * (blockIdx.x % horizBlocks);
    const int y = threadIdx.y + BLOCK_HEIGHT * (blockIdx.x / horizBlocks);
    const int t = blockIdx.y;

    if (x < dataWidth && y < dataHeight) {
	// compute the convolution
	if (type == SCALAR) {
	    float sum = 0;
	    for (int dt = -kernelFrames/2; dt <= kernelFrames/2; dt++) {
		for (int dy = -kernelHeight/2; dy <= kernelHeight/2; dy++) {
		    for (int dx = -kernelWidth/2; dx <= kernelWidth/2; dx++) {
			float weight = tex3D(texKernel, 
					     dx + kernelWidth/2, 
					     dy + kernelHeight/2, 
					     dt + kernelFrames/2);
			float value  = tex3D(texData,   
					     x + dx,
					     y + dy,
					     t + dt);
			sum += weight*value;
		    }
		}
	    }
	    output[((t*dataHeight + y)*dataWidth + x)*outputStride] = sum;	    
	} else if (type == INNER) {
	    float sum = 0;
	    for (int dt = -kernelFrames/2; dt <= kernelFrames/2; dt++) {
		for (int dy = -kernelHeight/2; dy <= kernelHeight/2; dy++) {
		    for (int dx = -kernelWidth/2; dx <= kernelWidth/2; dx++) {
			for (int c = 0; c < dataChannels; c++) {
			    float weight = tex3D(texKernel, 
						 (dx + kernelWidth/2)*kernelChannels + c, 
						 dy + kernelHeight/2, 
						 dt + kernelFrames/2);
			    float value  = tex3D(texData,   
						 (x + dx)*dataChannels + c,
						 y + dy,
						 t + dt);
			    sum += weight*value;
			}
		    }
		}
	    }
	    output[((t*dataHeight + y)*dataWidth + x)*outputStride] = sum;
	} else if (type == MATRIX) {
	    int outputChannels = kernelChannels / dataChannels;
	    for (int ck = 0; ck < kernelChannels; ck += dataChannels) {
		float sum = 0;
		for (int dt = -kernelFrames/2; dt <= kernelFrames/2; dt++) {
		    for (int dy = -kernelHeight/2; dy <= kernelHeight/2; dy++) {
			for (int dx = -kernelWidth/2; dx <= kernelWidth/2; dx++) {
			    for (int cd = 0; cd < dataChannels; cd++) {
				float weight = tex3D(texKernel, 
						     (dx + kernelWidth/2)*kernelChannels + ck + cd, 
						     dy + kernelHeight/2, 
						     dt + kernelFrames/2);
				float value  = tex3D(texData,   
						     (x + dx)*dataChannels + cd,
						     y + dy,
						     t + dt);
				sum += weight*value;
			    }
			}
		    }
		}
		output[(((t*dataHeight + y)*dataWidth + x)*outputChannels + ck/dataChannels)*outputStride] = sum;
	    }
	} else if (type == ELEMENTWISE) {
	    for (int c = 0; c < dataChannels; c++) {
		float sum = 0;
		for (int dt = -kernelFrames/2; dt <= kernelFrames/2; dt++) {
		    for (int dy = -kernelHeight/2; dy <= kernelHeight/2; dy++) {
			for (int dx = -kernelWidth/2; dx <= kernelWidth/2; dx++) {
			    float weight = tex3D(texKernel, 
						 (dx + kernelWidth/2)*kernelChannels + c, 
						 dy + kernelHeight/2, 
						 dt + kernelFrames/2);
			    float value  = tex3D(texData,   
						 (x + dx)*dataChannels + c,
						 y + dy,
						 t + dt);
			    sum += weight*value;
			}
		    }
		}
		output[(((t*dataHeight + y)*dataWidth + x)*dataChannels + c)*outputStride] = sum;
	    }
	} else { // (type == OUTER) 
	    for (int cd = 0; cd < dataChannels; cd++) {
		for (int ck = 0; ck < kernelChannels; ck++) {
		    float sum = 0;
		    for (int dt = -kernelFrames/2; dt <= kernelFrames/2; dt++) {
			for (int dy = -kernelHeight/2; dy <= kernelHeight/2; dy++) {
			    for (int dx = -kernelWidth/2; dx <= kernelWidth/2; dx++) {
				float weight = tex3D(texKernel, 
						     (dx + kernelWidth/2)*kernelChannels + ck, 
						     dy + kernelHeight/2, 
						     dt + kernelFrames/2);
				float value  = tex3D(texData,   
						     (x + dx)*dataChannels + cd,
						     y + dy,
						     t + dt);
				sum += weight*value;
			    }
			}
		    }
		    output[((((t*dataHeight + y)*dataWidth + x)*dataChannels + cd)*kernelChannels + ck)*outputStride] = sum;
		}
	    }
	}
    }    
}

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
}

void convolve3D(int dataFrames, int dataWidth, int dataHeight, int dataChannels, float *data, 
		int kernelFrames, int kernelWidth, int kernelHeight, int kernelChannels, float *kernel,
		float *output, convolve_t type, int outputStride) {

    cudaArray *deviceKernel, *deviceData;
    
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();   

    // check the args
    printf("type = %i (%i)\n", type, SCALAR);

    if (kernelChannels == 1 && dataChannels == 1) type = SCALAR;
    if (type == MATRIX && kernelChannels == dataChannels) type = INNER;
    if ((type == INNER || type == ELEMENTWISE) && 
	(dataChannels != kernelChannels)) {
	printf("For an inner or elementwise product, the kernel must have the same number of channels as the data\n");
	return;
    }
    if (type == MATRIX && (kernelChannels % dataChannels != 0)) {
	printf("For a matrix product, the kernel must have a number of channels that "
	       "is an integer multiple of the number of channels of the data.\n");
	return;
    }
    if (type == SCALAR && (dataChannels > 1 || kernelChannels > 1)) {
	printf("For a grayscale product, the kernel and data must each have one channel\n");
	return;	
    }
    if (outputStride < 1) {
	printf("Output stride must be at least one\n");
	return;
    }
    if (dataFrames < 1 || dataWidth < 1 || dataHeight < 1 || dataChannels < 1) {
	printf("All data dimensions must be at least one\n");
    }
    if (kernelFrames < 1 || kernelWidth < 1 || kernelHeight < 1 || kernelChannels < 1) {
	printf("All kernel dimensions must be at least one\n");
    }

    printf("Input data size           : %i x %i x %i x %i\n", 
	   dataFrames, dataWidth, dataHeight, dataChannels);
    printf("Convolution kernel size   : %i x %i x %i x %i\n",
	   kernelFrames, kernelWidth, kernelHeight, kernelChannels);

    printf("Allocating memory for output...\n");

    printf("Allocating and copying input data and convolution kernel from host to CUDA arrays...\n");

    struct cudaExtent dataExtent = make_cudaExtent(dataWidth*dataChannels, dataHeight, dataFrames);
    CUDA_SAFE_CALL(cudaMalloc3DArray(&deviceData, &channelDesc, dataExtent));
    struct cudaMemcpy3DParms dataCopyParms = {0};

    // first copy from host to device
    dataCopyParms.srcPtr = make_cudaPitchedPtr((void *)data, dataWidth*dataChannels*sizeof(float), 
					       dataWidth*dataChannels, dataHeight);
    dataCopyParms.dstArray = deviceData;
    dataCopyParms.extent = dataExtent;
    dataCopyParms.kind = cudaMemcpyHostToDevice;
    CUDA_SAFE_CALL(cudaMemcpy3D(&dataCopyParms));

    struct cudaExtent kernelExtent = {kernelWidth*kernelChannels, kernelHeight, kernelFrames};
    CUDA_SAFE_CALL(cudaMalloc3DArray(&deviceKernel, &channelDesc, kernelExtent));
    struct cudaMemcpy3DParms kernelCopyParms = {0};
    kernelCopyParms.srcPtr = make_cudaPitchedPtr((void *)kernel, kernelWidth*kernelChannels*sizeof(float), 
						 kernelWidth*kernelChannels, kernelHeight);
    kernelCopyParms.dstArray = deviceKernel;
    kernelCopyParms.extent = kernelExtent;
    kernelCopyParms.kind = cudaMemcpyHostToDevice;
    CUDA_SAFE_CALL(cudaMemcpy3D(&kernelCopyParms));

    printf("Binding CUDA arrays to texture references...\n");
    CUDA_SAFE_CALL(cudaBindTextureToArray(texKernel, deviceKernel));
    CUDA_SAFE_CALL(cudaBindTextureToArray(texData,   deviceData  ));

    dim3 threadBlock(BLOCK_WIDTH, BLOCK_HEIGHT);
    int horizBlocks = (dataWidth+BLOCK_WIDTH-1)/BLOCK_WIDTH;
    dim3 blockGrid((dataHeight+BLOCK_HEIGHT-1)/BLOCK_HEIGHT * horizBlocks, dataFrames);
    
    //unsigned int timer = 0;
    //cutCreateTimer(&timer);
    //cutResetTimer(timer);
    //cutStartTimer(timer);
    float *deviceOutput;
    int outputSize = outputStride*dataFrames*dataWidth*dataHeight*sizeof(float);
    switch(type) {
    case INNER: {
	CUDA_SAFE_CALL(cudaMalloc((void **)&deviceOutput, outputSize));
	_convolve3D<INNER><<<blockGrid, threadBlock>>>(dataWidth, dataHeight, dataChannels, horizBlocks,
						       kernelFrames, kernelWidth, kernelHeight, kernelChannels,
						       deviceOutput, outputStride);
	break;
    } 
    case OUTER: {
	outputSize *= dataChannels*kernelChannels;
	CUDA_SAFE_CALL(cudaMalloc((void **)&deviceOutput, outputSize));
	_convolve3D<OUTER><<<blockGrid, threadBlock>>>(dataWidth, dataHeight, dataChannels, horizBlocks,
						       kernelFrames, kernelWidth, kernelHeight, kernelChannels,
						       deviceOutput, outputStride);
	break;
    } 
    case MATRIX: {
	outputSize *= kernelChannels/dataChannels;
	CUDA_SAFE_CALL(cudaMalloc((void **)&deviceOutput, outputSize));
	_convolve3D<MATRIX><<<blockGrid, threadBlock>>>(dataWidth, dataHeight, dataChannels, horizBlocks,
							kernelFrames, kernelWidth, kernelHeight, kernelChannels,
							deviceOutput, outputStride);
	break;
    }
    case ELEMENTWISE: {
	outputSize *= dataChannels;
	CUDA_SAFE_CALL(cudaMalloc((void **)&deviceOutput, outputSize));
	_convolve3D<ELEMENTWISE><<<blockGrid, threadBlock>>>(dataWidth, dataHeight, dataChannels, horizBlocks,
							     kernelFrames, kernelWidth, kernelHeight, kernelChannels,
							     deviceOutput, outputStride);
	break;
    } 
    case SCALAR: {
	CUDA_SAFE_CALL(cudaMalloc((void **)&deviceOutput, outputSize));
	_convolve3D<SCALAR><<<blockGrid, threadBlock>>>(dataWidth, dataHeight, dataChannels, horizBlocks,
						      kernelFrames, kernelWidth, kernelHeight, kernelChannels,
						      deviceOutput, outputStride);
	break;	  
    }
    }

    //cutStopTimer(timer);
    CUT_CHECK_ERROR("convolve execution failed\n");
    
    //printf("Average time: %f\n", cutGetAverageTimerValue(timer));
    //cutDeleteTimer(timer);


    printf("Reading back results...\n");
    CUDA_SAFE_CALL(cudaMemcpy(output, deviceOutput, outputSize, cudaMemcpyDeviceToHost));    
    
    printf("Shutting down...\n");
    CUDA_SAFE_CALL(cudaUnbindTexture(texData));
    CUDA_SAFE_CALL(cudaUnbindTexture(texKernel));
    CUDA_SAFE_CALL(cudaFreeArray(deviceData));
    CUDA_SAFE_CALL(cudaFreeArray(deviceKernel));
    CUDA_SAFE_CALL(cudaFree(deviceOutput));
}
