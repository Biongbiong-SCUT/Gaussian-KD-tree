NVCC = /usr/local/cuda/bin/nvcc 

all: patchPCA filter

patchPCA: device.cu patchPCA.cpp device.h
	$(NVCC) device.cu patchPCA.cpp -o patchPCA -I ImageStack -L. -lImageStack -lcutil


filter: gpu_gkdtree.h gpu_gkdtree.cu gpu_gkdtree_build.cu gpu_gkdtree_include.cu gpu_gkdtree_finalize.cu gpu_gkdtree_splat.cu gpu_gkdtree_blur.cu gpu_gkdtree_slice.cu gpu_gkdtree_lookup.cu gpu_gkdtree_gaussian_lookup.cu gpu_gkdtree_util.cu filter.cpp
	$(NVCC) -arch sm_12 filter.cpp gpu_gkdtree.cu -o filter -I ImageStack -L. -lImageStack -lcutil


clean:
	rm filter patchPCA *.o *.linkinfo

