__global__ void checkValues(float *v, int d, int vCount, int *result) {
    const int myIdx = (blockIdx.x * 1024 + blockDim.y) * blockIdx.x + threadIdx.x;
    if (myIdx >= vCount) return;
    if (v[myIdx*d + d-1] <= 0.00000001) atomicAdd(result, 1);
    for (int i = 0; i < d; i++) {
	float val = v[myIdx*d+i];
	if (isnan(val)) atomicAdd(result+1, 1);
	if (isinf(val)) atomicAdd(result+2, 1);
    }
    
}

// blurring is just slicing on the leaves
void gaussianBlur(int samples, float sigma) {
    // grab the tree header
    kd_tree t;
    CUDA_SAFE_CALL(cudaMemcpy(&t, tree, sizeof(kd_tree), cudaMemcpyDeviceToHost));
    
    int roundedNPositions = ((t.leafCount-1)/LOOKUP_CHUNK_SIZE + 1);


    // copy the position array to the device
    float *newLeafValues;
    CUDA_SAFE_CALL(cudaMalloc((void **)&newLeafValues,
			      sizeof(float)*roundedNPositions*LOOKUP_CHUNK_SIZE*t.valueDimensions));


    int *_count;
    CUDA_SAFE_CALL(cudaMalloc((void **)&_count, 4*sizeof(int)));
    CUDA_SAFE_CALL(cudaMemset(_count, 0, 4*sizeof(int)));
    dim3 blockGrid((roundedNPositions-1)/1024+1, 1024);
    checkValues<<<blockGrid, LOOKUP_CHUNK_SIZE>>>(t.leafValues, t.valueDimensions, t.leafCount, _count);
    int count[4];
    CUDA_SAFE_CALL(cudaMemcpy(count, _count, 4*sizeof(int), cudaMemcpyDeviceToHost));
    printf("Check results: %i %i %i\n", count[0], count[1], count[2]);

    printf("Blurring"); fflush(stdout);

    // do 4k leaves at a time   
    int chunk = 1<<12;
    for (int i = 0; i < t.leafCount; i += chunk) {
	printf("."); fflush(stdout);
	if (i + chunk > t.leafCount) chunk = t.leafCount - i;
	int roundedChunk = ((chunk-1)/LOOKUP_CHUNK_SIZE + 1);
	uint4 seed = make_uint4(rand(), rand(), rand(), rand());
	_gaussianLookup<<<roundedChunk, LOOKUP_CHUNK_SIZE>>>(t, t.leafPositions+t.positionDimensions*i, 
							     newLeafValues+t.valueDimensions*i, 
							     chunk, sigma, samples, seed, SLICE);
	CUT_CHECK_ERROR("gaussianSlice failed\n");
    }

    printf("Done\n"); fflush(stdout);


    CUDA_SAFE_CALL(cudaMemset(_count, 0, 4*sizeof(int)));
    checkValues<<<blockGrid, LOOKUP_CHUNK_SIZE>>>(newLeafValues, t.valueDimensions, t.leafCount, _count);
    CUDA_SAFE_CALL(cudaMemcpy(count, _count, 4*sizeof(int), cudaMemcpyDeviceToHost));
    printf("Check results: %i %i %i\n", count[0], count[1], count[2]);
    CUDA_SAFE_CALL(cudaFree(_count));

    CUDA_SAFE_CALL(cudaFree(t.leafValues));
    t.leafValues = newLeafValues;
    leafValues = newLeafValues;

    CUDA_SAFE_CALL(cudaMemcpy(tree, &t, sizeof(kd_tree), cudaMemcpyHostToDevice));
}

