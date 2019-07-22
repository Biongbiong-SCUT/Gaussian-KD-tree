// we pass the tree by value, because we need to load most of it anyway

void gaussianSlice(float *positions, float *values, 
		   int nPositions, int samples, float sigma) {
    
    // grab the tree header
    kd_tree t;
    CUDA_SAFE_CALL(cudaMemcpy(&t, tree, sizeof(kd_tree), cudaMemcpyDeviceToHost));
    
    int roundedNPositions = ((nPositions-1)/LOOKUP_CHUNK_SIZE+1)*LOOKUP_CHUNK_SIZE;

    // copy the position array to the device
    float *devicePositions;
    CUDA_SAFE_CALL(cudaMalloc((void **)&devicePositions,
			      sizeof(float)*roundedNPositions*t.positionDimensions));
    CUDA_SAFE_CALL(cudaMemcpy(devicePositions, positions,
			      sizeof(float)*nPositions*t.positionDimensions,
			      cudaMemcpyHostToDevice));

    // clear the debug array
    CUDA_SAFE_CALL(cudaMemset(debugPtr, 0, sizeof(int)*1024*1024));
    
    // allocate the values array on the device
    float *deviceValues;
    CUDA_SAFE_CALL(cudaMalloc((void **)&deviceValues, 
			      sizeof(float)*nPositions*t.valueDimensions));
    
    printf("Slicing... "); fflush(stdout);
    // do 4k samples at a time   
    int chunk = 1<<12;
    for (int i = 0; i < nPositions; i += chunk) {
	printf("."); fflush(stdout);
	if (i + chunk > nPositions) chunk = nPositions - i;
	int roundedChunk = ((chunk-1)/LOOKUP_CHUNK_SIZE + 1);
	uint4 seed = make_uint4(rand(), rand(), rand(), rand());
	_gaussianLookup<<<roundedChunk, LOOKUP_CHUNK_SIZE>>>(t, devicePositions + i*t.positionDimensions,
							     deviceValues + i*t.valueDimensions,
							     chunk, sigma, samples, seed, SLICE);
	CUT_CHECK_ERROR("gaussianLookup failed\n");
    }

    printf("Done\n"); fflush(stdout);

    // copy back the values
    CUDA_SAFE_CALL(cudaMemcpy(values, deviceValues, 
			      sizeof(float)*nPositions*t.valueDimensions,
			      cudaMemcpyDeviceToHost));
    
    // check the debug array
    /*
    int dbg[1024*20];
    CUDA_SAFE_CALL(cudaMemcpy(dbg, debugPtr, sizeof(int)*1024, cudaMemcpyDeviceToHost));
    for (int i = 0; i < 1024*20; i+=4) {
	printf("%7i \t%7i \t%7i \t%7i\n", dbg[i], dbg[i+1], dbg[i+2], dbg[i+3]);
    }
    fflush(stdout);
    */

    // clean up
    CUDA_SAFE_CALL(cudaFree(devicePositions));
    CUDA_SAFE_CALL(cudaFree(deviceValues));
}

