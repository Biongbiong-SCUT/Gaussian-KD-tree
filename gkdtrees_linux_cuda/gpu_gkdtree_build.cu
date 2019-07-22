
struct build_job {
    // take this data
    float *data;
    int nData;
    // either compute the best split, allocate a split node, and make
    // owner[0] = the appropriate index in the internal node array or
    // decide it's a leaf node, and make -owner[0] = the appropriate
    // index in the leaf array
    int *owner;
    int parent;

};

struct build_job_queue {
    build_job *in;
    int inCount;
    build_job *out;
    int outCount;
};

static __global__ void _manageQueue(build_job_queue *queue, int doneJobs);
static __global__ void _buildTree(kd_tree *tree, build_job_queue *queue);

void buildLargeTree(kd_tree *tree, build_job_queue *queue, build_job job);

void buildTree(int posDimensions, int valDimensions, float *data, int nData, int nAllocated) {
    // test the random number generator
    //urandTest<<<1, 1>>>();

    srand(time(NULL));

    // account for the weight dimension
    valDimensions++;

    maxLeaves = nAllocated;

    // copy the data to the gpu
    float *deviceData;
    CUDA_SAFE_CALL(cudaMalloc((void **)&deviceData, nData*posDimensions*sizeof(float)));
    CUDA_SAFE_CALL(cudaMemcpy(deviceData, data, nData*posDimensions*sizeof(float), cudaMemcpyHostToDevice));

    kd_tree t;
    t.positionDimensions = posDimensions;
    t.valueDimensions = valDimensions;

    printf("Allocating %i MB for node storage\n", 
	   (nAllocated*sizeof(node) + nAllocated*posDimensions*sizeof(float)) >> 20);
    fflush(stdout);

    // allocate space for the internal nodes
    CUDA_SAFE_CALL(cudaMalloc((void **)&t.nodeArray, nAllocated*sizeof(node)));

    // allocate space for the leaves    
    CUDA_SAFE_CALL(cudaMalloc((void **)&t.leafPositions, nAllocated*posDimensions*sizeof(float)));
    nodeArray = t.nodeArray;
    leafPositions = t.leafPositions;

    // allocate space for the work queues
    build_job_queue q = {NULL, 0, NULL, 0};
    CUDA_SAFE_CALL(cudaMalloc((void **)&q.in, nData*sizeof(build_job)));
    CUDA_SAFE_CALL(cudaMalloc((void **)&q.out, nData*sizeof(build_job)));   

    t.nodeCount = t.leafCount = 0;

    // start off the first work queue with a job to build the root node
    build_job build_root = {deviceData, nData, NULL, -1};
    //CUDA_SAFE_CALL(cudaMemcpy(q.out, &build_root, sizeof(build_job), cudaMemcpyHostToDevice));

    // build the root portion of the tree
    buildLargeTree(&t, &q, build_root);

    // copy the tree header to the device
    CUDA_SAFE_CALL(cudaMalloc((void **)&tree, sizeof(kd_tree)));
    CUDA_SAFE_CALL(cudaMemcpy(tree, &t, sizeof(kd_tree), cudaMemcpyHostToDevice));

    // copy the queue header to the device
    build_job_queue *queue;
    CUDA_SAFE_CALL(cudaMalloc((void **)&queue, sizeof(build_job_queue)));
    CUDA_SAFE_CALL(cudaMemcpy(queue, &q, sizeof(build_job_queue), cudaMemcpyHostToDevice));

    int total = 0;

    //for (int i = 0; i < 20; i++) {
    while (q.inCount + q.outCount > 0) {
	dim3 threadBlock(posDimensions, 1);
	dim3 blockGrid(MAX_BUILD_BLOCKS, 1);

	_buildTree<<<blockGrid, threadBlock>>>(tree, queue);
	CUT_CHECK_ERROR("buildTree failed\n");
	fflush(stdout);

	/*
	int localDebug[1024];
	CUDA_SAFE_CALL(cudaMemcpy(localDebug, debugPtr, sizeof(int)*1024, cudaMemcpyDeviceToHost));
	printf("Debug: ");
	for (int j = 0; j < 256; j++) {
	    printf("%i ", localDebug[j]);
	}
	printf("\n");
	*/

	CUDA_SAFE_CALL(cudaMemcpy(&q, queue, sizeof(build_job_queue), cudaMemcpyDeviceToHost));

	if (q.outCount > MAX_BUILD_BLOCKS) total += MAX_BUILD_BLOCKS;
	else total += q.outCount;

	_manageQueue<<<1, 1>>>(queue, MAX_BUILD_BLOCKS);
	CUT_CHECK_ERROR("manageQueue failed\n");
	fflush(stdout);

	CUDA_SAFE_CALL(cudaMemcpy(&q, queue, sizeof(build_job_queue), cudaMemcpyDeviceToHost));
	printf("%i jobs performed\n", total);
    }

    // copy the tree header from device to host
    CUDA_SAFE_CALL(cudaMemcpy(&t, tree, sizeof(kd_tree), cudaMemcpyDeviceToHost));

    // allocate space for leaf values
    CUDA_SAFE_CALL(cudaMalloc((void **)&t.leafValues, sizeof(float)*valDimensions*nAllocated));
    leafValues = t.leafValues;

    // set them to zero
    CUDA_SAFE_CALL(cudaMemset(t.leafValues, 0, sizeof(float)*valDimensions*nAllocated));

    // copy the tree header back to the device
    CUDA_SAFE_CALL(cudaMemcpy(tree, &t, sizeof(kd_tree), cudaMemcpyHostToDevice));

    fflush(stdout);

    printf("Constructed a tree with %i leaves\n", t.leafCount); fflush(stdout);

    // free the data
    CUDA_SAFE_CALL(cudaFree(deviceData));

    // free the work queues
    CUDA_SAFE_CALL(cudaFree(q.in));
    CUDA_SAFE_CALL(cudaFree(q.out));
    CUDA_SAFE_CALL(cudaFree(queue));
}

void destroyTree() {
    // free the internal nodes
    CUDA_SAFE_CALL(cudaFree(nodeArray));

    // free the leaves
    CUDA_SAFE_CALL(cudaFree(leafPositions));
    CUDA_SAFE_CALL(cudaFree(leafValues));

    // free the header
    CUDA_SAFE_CALL(cudaFree(tree));
}

static __global__ void _manageQueue(build_job_queue *q, int doneJobs) {
    if (blockIdx.x != 0) return;
    if (threadIdx.x != 0) return;

    // up to 'doneJobs' jobs were just taken from the end of the out queue
    DEBUG(printf("counts: %i %i\n", q->outCount, q->inCount));

    q->outCount -= doneJobs;
    
    if (q->outCount <= 0) {
	q->outCount = q->inCount;
	q->inCount = 0;
	build_job *j = q->in;
	q->in = q->out;
	q->out = j;
    }
}

static __device__ void _buildSplitNode(float *boundingBox, float *data, 
				       int dimensions, int nData, node *n, int *splitIdx);
static __device__ void _computeBoundingBox(float *data, int dimensions, int nData, float *bbox);

static __global__ void _buildTree(kd_tree *tree, build_job_queue *queue) {
    int dimensions = tree->positionDimensions;
    
    int myJobIdx = queue->outCount - blockIdx.x - 1;

    if (myJobIdx < 0) return;

    DEBUG0(printf("My job index is %i/%i\n", myJobIdx, queue->outCount));
    
    // pop from the work queue
    __shared__ build_job myJob;
    if (threadIdx.x == 0) {
	myJob = queue->out[myJobIdx];   

	DEBUG(printf("Popped a job off the queue\n", myJobIdx, queue->outCount));
    }
    __syncthreads();
    
    
    // compute bounding box and diagonal length
    __shared__ float diagonal;

    __shared__ float boundingBox[MAX_DIMENSIONS*2];    

    DEBUG0(printf("Computing bounding box %x\n", myJob.data));
    _computeBoundingBox(myJob.data, dimensions, myJob.nData, boundingBox);
    DEBUG0(printf("Done computing bounding box\n"));
	      
    float myLength = boundingBox[threadIdx.x*2+1];
    myLength *= myLength;

    for (int i = 0; i < dimensions; i++) {
	if (i == threadIdx.x) {
	    if (threadIdx.x == 0) diagonal = myLength;
	    else diagonal += myLength;
	}
	__syncthreads();
    }

    DEBUG0(printf("Diagonal length is %f\n", diagonal));

    if (diagonal > 4*BUILD_SIZE_LIMIT*BUILD_SIZE_LIMIT) {
	__shared__ node *mySplitNode;
	__shared__ int mySplitNodeIdx;

	DEBUG0(printf("nodeArray = %x\n", tree->nodeArray));

	// allocate a new split node
	if (threadIdx.x == 0) {
	    mySplitNodeIdx = atomicAdd(&tree->nodeCount, 1);
	    mySplitNode = tree->nodeArray + mySplitNodeIdx;
	    if (myJob.owner != NULL) myJob.owner[0] = mySplitNodeIdx;
	}
	__syncthreads();

	__shared__ int splitIdx;

	DEBUG0(printf("Calling build split node with args: %x %i %i %x %x\n", 
			myJob.data, dimensions, myJob.nData, mySplitNode, &splitIdx));


	_buildSplitNode(boundingBox, myJob.data, dimensions, myJob.nData, mySplitNode, &splitIdx);

	// make two new jobs
	if (threadIdx.x == 0) {
	    mySplitNode->parent = myJob.parent;

	    DEBUG(printf("Back from building a split, now allocating new jobs. One job is size %i, the other is size %i.\n", splitIdx, myJob.nData-splitIdx));

	    int n = atomicAdd(&(queue->inCount), 2);
	    DEBUG(
		printf("Jobs go in slots %i, %i\n", n, n+1);
		if (splitIdx <= 0 || splitIdx >= myJob.nData) {
		    printf("ERROR: split index out of bounds %i %i\n", splitIdx, myJob.nData);
		}
		);

	    queue->in[n+1].data       = myJob.data;
	    queue->in[n+1].nData      = splitIdx;
	    queue->in[n+1].owner      = &(mySplitNode->left);
	    queue->in[n+1].parent     = mySplitNodeIdx;
	    queue->in[n].data         = myJob.data + splitIdx * dimensions;
	    queue->in[n].nData        = myJob.nData - splitIdx;
	    queue->in[n].owner        = &(mySplitNode->right);
	    queue->in[n].parent       = mySplitNodeIdx;
	}
	
    } else {
	// build a leaf node

	DEBUG0(printf("Making a leaf out of %i data elements\n", myJob.nData));

	volatile __shared__ float *myLeafNode;

	if (threadIdx.x == 0) {
	    int n = atomicAdd(&tree->leafCount, 1);
	    myLeafNode = tree->leafPositions + n*dimensions;
	    if (myJob.owner != NULL) myJob.owner[0] = -n;
	}
	__syncthreads();

	DEBUG0(printf("Doing the averaging\n"));

	float val = 0;
	float *dataPtr = myJob.data;
	for (int i = 0; i < myJob.nData; i++) {
	    val += dataPtr[threadIdx.x];
	    dataPtr += dimensions;
	}
	val /= myJob.nData;

	myLeafNode[threadIdx.x] = val;

	DEBUG0(printf("Done making leaf\n"));
    }
}

static __device__ void _computeBoundingBox(float *data, int dimensions, int nData,
					   float *boundingBox) {

    // compute a bounding box
    float *dataPtr = data + threadIdx.x;
    float val = *dataPtr;
    float *bbox = boundingBox + 2*threadIdx.x;
    dataPtr += dimensions;
    
    bbox[0] = val;
    bbox[1] = val;
    
    // TODO: use atomic min and atomic max and parallelize more (int is fine)
    for (int i = 1; i < nData; i++) {
	val = *dataPtr;
	dataPtr += dimensions;
	bbox[0] = fminf(bbox[0], val);
	bbox[1] = fmaxf(bbox[1], val);
    }
    
    // convert from min, max to min, size
    bbox[1] -= bbox[0];

    __syncthreads();
}


static __device__ void _buildSplitNode(float *boundingBox, float *data, 
				       int dimensions, int nData, node *n, int *splitIdx) {

    // figure out the longest dimension in the bounding box
    __shared__ int longest;
    __shared__ float length;

    if (threadIdx.x == 0) {
	length = 0;
	longest = 0;
    }

    float myLength = boundingBox[threadIdx.x*2+1];

    __syncthreads();

    // decide on the length of the longest dim
    atomicMax((int *)(&length), *((int *)(&myLength)));

    __syncthreads();
    
    // decide on the longest dim
    if (length == myLength) {
	atomicMax(&longest, threadIdx.x);
    } 
    
    __syncthreads();

    __shared__ float fsplit;

    DEBUG0(printf("Cutting the node in half\n"));
    if (threadIdx.x == 0) {
	fsplit = boundingBox[longest*2] + boundingBox[longest*2+1]/2;
    }
    __syncthreads();

    if (threadIdx.x == 0) {
	DEBUG(printf("Writing to n (%x)\n", n));
	n->cut_dim = longest;
	n->cut_val = fsplit;
    }

    DEBUG0(printf("Starting sort\n"));

    // now split the data over the split point
    
    volatile __shared__ int start, end;
    
    if (threadIdx.x == longest) {
	start = -dimensions;
	end = nData*dimensions;

	/*
	DEBUG({
	    printf("Data before sort:\n");
	    for (int i = 0; i < nData; i++) {
		printf("%f\n", data[i*dimensions + longest]);
	    }
	    fflush(stdout);
	    
	    });
	*/
    }

    // the limit on the loop isn't strictly necessary, but if I
    // introduce a bug in the loop it prevents it from looping
    // infinitely and crashing the computer due to an unresponsive GPU
    for (int i = 0; i < nData; i++) {
	
	if (threadIdx.x == longest) {
	    start += dimensions;
	    end -= dimensions;

	    // walk from the bottom up until we find the first value
	    // that should be in the second half 	    
	    while (data[start + longest] <= fsplit && start < end) start += dimensions;
	    
	    // walk from the top down until we find the first value
	    // that should be in the first half
	    while (data[end + longest] > fsplit && end > start) end -= dimensions;
       
	    DEBUG(printf("split %i %i (%i, %i)\n", 
			 start/dimensions, end/dimensions, nData, i));


	}	

	__syncthreads();

	if (start >= end) {
	    break;
	}

	// all threads swap start and end
	float tmp = data[start+threadIdx.x];
	data[start+threadIdx.x] = data[end+threadIdx.x];
	data[end+threadIdx.x] = tmp;
	
	__syncthreads();

    }
    __syncthreads();

    if (threadIdx.x == longest) {

	if (data[start+longest] <= fsplit) start += dimensions;
	    
	/*
	DEBUG({
	    printf("Data after sort:\n");
	    for (int i = 0; i < nData; i++) {
		if (i == start/dimensions) {
		    printf("SPLIT\n");
		}
		printf("%f\n", data[i*dimensions + longest]);
	    }
	    fflush(stdout);
	    });
	*/

	//splitIdx[0] = nData/2;
	splitIdx[0] = start/dimensions;
	DEBUG(printf("split index = %i / %i\n", start/dimensions, nData));	
	DEBUG(printf("Split val: %f\n", fsplit));
	DEBUG(printf("Values surrounding split: %f %f\n", 
		     data[start+longest-dimensions], data[start+longest]));


	//for (int i = 0; i < nData; i++) {
	//debug[i] = data[i*dimensions+longest];
	//}
    }



    __syncthreads();
    
    DEBUG0(printf("Done building node\n"));
}


// Stuff for building the root nodes below here
static __global__ void _computeLargeBoundingBox(float *data, int nData, float *bBox);
static __global__ void _countSmaller(float *data, int nData, int splitDim, int dims, float splitVal, int *count);
static __global__ void _computeSwapLists(float *data, int nData, int splitDim, int dims, 
					 float splitVal, int count, int *swapList, int *swapCount);
static __global__ void _performSwaps(float *data, int *swapList, int swapCount);


void buildLargeTree(kd_tree *tree, build_job_queue *queue, build_job job) {
    if (job.nData <= 1) {
	printf("Leaf node during root building phase!\n"); fflush(stdout);
	exit(1);
    } else if (job.nData < (1<<20)) {
	printf("Small job (%i), deferring to GPU\n", job.nData); fflush(stdout);
	// it's a small job, make a single GPU block do all of it	
	CUDA_SAFE_CALL(cudaMemcpy(queue->out + queue->outCount, &job, 
				  sizeof(build_job), cudaMemcpyHostToDevice));
	queue->outCount++;
    } else {
	printf("Performing job of size %i\n", job.nData); fflush(stdout);
	// compute a bounding box
	float *bBox = new float[tree->positionDimensions*2];
	for (int i = 0; i < tree->positionDimensions; i++) {
	    bBox[i*2] = INF;
	    bBox[i*2+1] = -INF;
	}

	float *deviceBBox;
	CUDA_SAFE_CALL(cudaMalloc((void **)&deviceBBox, tree->positionDimensions*2*sizeof(float)));
	CUDA_SAFE_CALL(cudaMemcpy(deviceBBox, bBox,
				  tree->positionDimensions*2*sizeof(float),
				  cudaMemcpyHostToDevice));
	int blocks = (job.nData - 1 / 256) + 1;
	if (blocks > 1<<15) blocks = 1<<15;
	dim3 blockDim(blocks, tree->positionDimensions);
	dim3 threadDim(256, 1);
	_computeLargeBoundingBox<<<blockDim, threadDim>>>(job.data, job.nData, deviceBBox);
	CUT_CHECK_ERROR("computeLargeBoundingBox failed\n");
	CUDA_SAFE_CALL(cudaMemcpy(bBox, deviceBBox, 
				  tree->positionDimensions*2*sizeof(float), 
				  cudaMemcpyDeviceToHost));	
	CUDA_SAFE_CALL(cudaFree(deviceBBox));

	int longest;
	float length = 0;
	printf("Bounding box: \n");
	for (int i = 0; i < tree->positionDimensions; i++) {
	    printf(" %f %f\n", bBox[i*2], bBox[i*2+1]);
	    float l = bBox[i*2+1] - bBox[i*2];
	    if (l > length) {
		length = l;
		longest = i;
	    }
	}
	fflush(stdout);
	float split = (bBox[longest*2+1] + bBox[longest*2])/2;
	delete bBox;

	printf("Building a split node: %i %f\n", longest, split); fflush(stdout);

	// now split the data over the longest dimension
	int *deviceSplitPoint = NULL;
	int splitPoint = 0;
	CUDA_SAFE_CALL(cudaMalloc((void **)&deviceSplitPoint, sizeof(int)));
	CUDA_SAFE_CALL(cudaMemcpy(deviceSplitPoint, &splitPoint, sizeof(int), cudaMemcpyHostToDevice));
	_countSmaller<<<blocks, threadDim>>>(job.data, job.nData, longest, 
					     tree->positionDimensions, split, deviceSplitPoint);
	CUT_CHECK_ERROR("countSmaller failed\n");


	CUDA_SAFE_CALL(cudaMemcpy(&splitPoint, deviceSplitPoint, sizeof(int), cudaMemcpyDeviceToHost));
	printf("%i/%i values were smaller than the split\n", splitPoint, job.nData); fflush(stdout);
	CUDA_SAFE_CALL(cudaFree(deviceSplitPoint));
	
	// make a pair of lists of all vectors that need to be moved over the split
	int *swapList = NULL, *swapCount = NULL;
	CUDA_SAFE_CALL(cudaMalloc((void **)&swapList, sizeof(int)*job.nData));
	CUDA_SAFE_CALL(cudaMalloc((void **)&swapCount, sizeof(int)*2));
	CUDA_SAFE_CALL(cudaMemset(swapCount, 0, sizeof(int)*2));
	_computeSwapLists<<<blocks, threadDim>>>(job.data, job.nData, longest, tree->positionDimensions, split, splitPoint, swapList, swapCount);
	CUT_CHECK_ERROR("computeSwapLists failed\n");	

	int hostSwapCount[2];
	CUDA_SAFE_CALL(cudaMemcpy(hostSwapCount, swapCount, sizeof(int)*2, cudaMemcpyDeviceToHost));
	printf("%i %i elements in the swap lists\n", hostSwapCount[0], hostSwapCount[1]); fflush(stdout);
	CUDA_SAFE_CALL(cudaFree(swapCount));

	if (blockDim.x > hostSwapCount[0]) blockDim.x = hostSwapCount[0];
	if (blockDim.x > 0) _performSwaps<<<blockDim, threadDim>>>(job.data, swapList, hostSwapCount[0]);
	CUT_CHECK_ERROR("performSwaps failed\n"); 

	CUDA_SAFE_CALL(cudaFree(swapList));

	// now allocate the split node
	node n;
	n.cut_dim = longest;
	n.cut_val = split;
	n.parent = job.parent;
	n.left = 0;
	n.right = 0;
	// put it on the device
	CUDA_SAFE_CALL(cudaMemcpy(tree->nodeArray + tree->nodeCount, &n, sizeof(node), cudaMemcpyHostToDevice));

	// set the parent to point to it
	if (job.owner != NULL) {
	    printf("Setting owner pointer at %x to %i\n", job.owner, tree->nodeCount);
	    CUDA_SAFE_CALL(cudaMemcpy(job.owner, &tree->nodeCount, sizeof(int), cudaMemcpyHostToDevice));
	}

	int parent = tree->nodeCount;

	tree->nodeCount++;

	// make two child jobs
	build_job child;
	child.data = job.data;
	child.nData = splitPoint;
	child.owner = &(tree->nodeArray[parent].left);
	child.parent = parent;
	buildLargeTree(tree, queue, child);

	child.data = job.data + splitPoint * tree->positionDimensions;
	child.nData = job.nData - splitPoint;
	child.owner = &(tree->nodeArray[parent].right);
	buildLargeTree(tree, queue, child);

    }
}


static __global__ void _computeLargeBoundingBox(float *data, int nData, float *bBox) {
    const int myIdx = blockIdx.x*blockDim.x + threadIdx.x;

    __shared__ float min, max;

    if (threadIdx.x == 0) {
	min = INF;
	max = -INF;
    }
    __syncthreads();

    for (int i = myIdx; i < nData; i += gridDim.x * blockDim.x) {
	DEBUG(printf("testing vector %i dimension %i (%i)\n", i, blockIdx.y, gridDim.y));
	

	float val = data[i*gridDim.y + blockIdx.y];
	atomicMin((int *)&min, __float_as_int(val));
	atomicMax((int *)&max, __float_as_int(val));
    }

    __syncthreads();

    if (threadIdx.x == 0) {
	atomicMin((int *)(bBox + blockIdx.y*2), __float_as_int(min));
	atomicMax((int *)(bBox + blockIdx.y*2+1), __float_as_int(max));
    }
}
static __global__ void _countSmaller(float *data, int nData, int splitDim, 
				     int dims, float splitVal, int *count) {
    const int myIdx = blockIdx.x*blockDim.x + threadIdx.x;

    __shared__ int sharedCount;
    if (threadIdx.x == 0) sharedCount = 0;
    __syncthreads();

    int localCount = 0;
    for (int i = myIdx; i < nData; i += gridDim.x * blockDim.x) {
	if (data[i*dims + splitDim] <= splitVal) 
	    localCount++;
    }
    
    atomicAdd(&sharedCount, localCount);

    __syncthreads();
    if (threadIdx.x == 0) atomicAdd(count, sharedCount);
}

static __global__ void _computeSwapLists(float *data, int nData, int splitDim, int dims, 
					 float splitVal, int count, int *swapList, int *swapCount) {
    const int myIdx = blockIdx.x*blockDim.x + threadIdx.x;

    for (int i = myIdx; i < nData; i += gridDim.x * blockDim.x) {
	float val = data[i*dims + splitDim];
	if (val <= splitVal && i >= count) {
	    int pos = atomicAdd(swapCount, 1);
	    swapList[pos*2] = i;
	} else if (val > splitVal && i < count) {
	    int pos = atomicAdd(swapCount+1, 1);
	    swapList[pos*2+1] = i;
	}
    }
	
}

static __global__ void _performSwaps(float *data, int *swapList, int swapCount) {
    const int myIdx = blockIdx.x*blockDim.x + threadIdx.x;

    for (int i = myIdx; i < swapCount; i += gridDim.x * blockDim.x) {
	int a = swapList[i*2];
	int b = swapList[i*2+1];
	float val = data[a * gridDim.y + blockIdx.y];
	data[a * gridDim.y + blockIdx.y] = data[b * gridDim.y + blockIdx.y];
	data[b * gridDim.y + blockIdx.y] = val;	
    }
}

