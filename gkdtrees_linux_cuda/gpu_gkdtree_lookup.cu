
#define CHUNK_SIZE 16

static __global__ void _lookup(kd_tree *tree, float *query, int dimensions, int *leaves);

void lookup(float *query, int dimensions, int nQuery, int *leaves) {
    // round up nQuery to the nearest multiply of CHUNK_SIZE
    int roundedNQuery = ((nQuery-1)/CHUNK_SIZE + 1)*CHUNK_SIZE;

    // allocate space for the query vector
    float *deviceQuery;
    CUDA_SAFE_CALL(cudaMalloc((void **)&deviceQuery, roundedNQuery*dimensions*sizeof(float)));    
    CUDA_SAFE_CALL(cudaMemcpy(deviceQuery, query, nQuery*dimensions*sizeof(float), cudaMemcpyHostToDevice));

    // allocate space for the response vector
    int *deviceLeaves;
    CUDA_SAFE_CALL(cudaMalloc((void **)&deviceLeaves, roundedNQuery*sizeof(int)));

    dim3 threadBlock(CHUNK_SIZE, 1);
    dim3 blockGrid((nQuery-1)/CHUNK_SIZE+1, 1);

    _lookup<<<blockGrid, threadBlock>>>(tree, deviceQuery, dimensions, deviceLeaves);        

    // copy back response
    CUDA_SAFE_CALL(cudaMemcpy(leaves, deviceLeaves, nQuery*sizeof(int), cudaMemcpyDeviceToHost));

    CUDA_SAFE_CALL(cudaFree(deviceQuery));
    CUDA_SAFE_CALL(cudaFree(deviceLeaves));

    fflush(stdout);
}

static __global__ void _lookup(kd_tree *tree, float *query, int dimensions, int *leaves) {
    float *queryBlock = query + blockIdx.x * CHUNK_SIZE * dimensions;

    // put my vector in shared memory
    __shared__ float localQuery[CHUNK_SIZE*MAX_DIMENSIONS];
    for (int i = threadIdx.x; i < dimensions*CHUNK_SIZE; i += CHUNK_SIZE) {
	localQuery[i] = queryBlock[i];
    }

    float *myQuery = localQuery + threadIdx.x * dimensions;

    __syncthreads();

    // ok, we have my query, we're good to go
    int nodeIdx = 0;
    node *nodeArray = tree->nodeArray;
    do {
	node n = nodeArray[nodeIdx];	
	DEBUG(printf("Retrieved a node: %i, %i, %f, %i, %i\n", nodeIdx, n.cut_dim, n.cut_val, n.left, n.right));
	DEBUG(printf("Query %i with value %f\n",
		     blockIdx.x * CHUNK_SIZE + threadIdx.x, myQuery[n.cut_dim]));
	if (myQuery[n.cut_dim] <= n.cut_val) {
	    DEBUG(printf("Query %i splitting left to node %i\n", blockIdx.x * CHUNK_SIZE + threadIdx.x, n.left));
	    nodeIdx = n.left;
	} else {
	    DEBUG(printf("Query %i splitting right to node %i\n", blockIdx.x * CHUNK_SIZE + threadIdx.x, n.right));
	    nodeIdx = n.right;	
	}
    } while (nodeIdx > 0);

    // we're at a leaf node, put it's ID in the leaves array
    leaves[blockIdx.x * CHUNK_SIZE + threadIdx.x] = -nodeIdx;    
}
