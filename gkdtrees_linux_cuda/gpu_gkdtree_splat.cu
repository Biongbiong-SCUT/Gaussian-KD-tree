

void gaussianSplat(float *positions, float *values, 
		   int nPositions, int samples, float sigma) {
    
    int roundedNPositions = ((nPositions-1)/LOOKUP_CHUNK_SIZE + 1)*LOOKUP_CHUNK_SIZE;
    
    // grab the tree header
    kd_tree t;
    CUDA_SAFE_CALL(cudaMemcpy(&t, tree, sizeof(kd_tree), cudaMemcpyDeviceToHost));
    
    // copy the position array to the device
    float *devicePositions;
    CUDA_SAFE_CALL(cudaMalloc((void **)&devicePositions,
			      sizeof(float)*roundedNPositions*t.positionDimensions));
    CUDA_SAFE_CALL(cudaMemcpy(devicePositions, positions,
			      sizeof(float)*nPositions*t.positionDimensions,
			      cudaMemcpyHostToDevice));
    
    // copy the values array to the device
    float *deviceValues;
    CUDA_SAFE_CALL(cudaMalloc((void **)&deviceValues, 
			      sizeof(float)*roundedNPositions*(t.valueDimensions-1)));
    CUDA_SAFE_CALL(cudaMemcpy(deviceValues, values, 
			      sizeof(float)*nPositions*(t.valueDimensions-1),
			      cudaMemcpyHostToDevice));

    printf("Splatting... "); fflush(stdout);

    // do 4k samples at a time   
    int chunk = 1<<12;
    for (int i = 0; i < nPositions; i += chunk) {
	printf("."); fflush(stdout);
	if (i + chunk > nPositions) chunk = nPositions - i;
	int roundedChunk = ((chunk-1)/LOOKUP_CHUNK_SIZE + 1);
	uint4 seed = make_uint4(rand(), rand(), rand(), rand());
	_gaussianLookup<<<roundedChunk, LOOKUP_CHUNK_SIZE>>>(t, devicePositions + i*t.positionDimensions,
							     deviceValues + i*(t.valueDimensions-1),
							     chunk, sigma, samples, seed, SPLAT);
	CUT_CHECK_ERROR("gaussianLookup failed\n");
    }
   
    printf("Done\n"); fflush(stdout);
		   
    // clean up
    CUDA_SAFE_CALL(cudaFree(devicePositions));
    CUDA_SAFE_CALL(cudaFree(deviceValues));
}


