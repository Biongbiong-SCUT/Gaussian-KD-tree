struct include_hit {
    int parent; // the sign bit indicates left or right
    int next;
};

#define INCLUDE_CHUNK_SIZE 64

static __global__ void _includeLookup(kd_tree *tree, float *positions, int nPositions, 
				      include_hit *hits, int *hitLists);

static __global__ void _reduceIncludeHitLists(kd_tree *tree, int oldLeafCount, float *positions, 
					      include_hit *hits, int *hitLists);

void include(float *positions, int nPositions) {
    // grab the tree header
    kd_tree t;
    CUDA_SAFE_CALL(cudaMemcpy(&t, tree, sizeof(kd_tree), cudaMemcpyDeviceToHost));    

    // check if there's a danger of running out of memory
    if (nPositions + t.leafCount > maxLeaves) {
	printf("ERROR: Cannot include this many nodes! The tree already has %i leaves, there "
	       "is only room for %i, and this call would potentially add %i more "
	       "leaves bringing the total to %i.\n", 
	       t.leafCount, maxLeaves, nPositions, nPositions + t.leafCount);
	fflush(stdout);
	exit(1);
	return;
    }

    // copy the new nodes to the device
    float *devicePositions;
    CUDA_SAFE_CALL(cudaMalloc((void **)&devicePositions, sizeof(float)*nPositions*t.positionDimensions));
    CUDA_SAFE_CALL(cudaMemcpy(devicePositions, positions, sizeof(float)*nPositions*t.positionDimensions, cudaMemcpyHostToDevice));

    // make the hit array
    include_hit *includeHits;
    CUDA_SAFE_CALL(cudaMalloc((void **)&includeHits, sizeof(include_hit)*nPositions));
    CUDA_SAFE_CALL(cudaMemset(includeHits, -1, sizeof(include_hit)*nPositions));

    // make the lists
    int *hitLists;
    CUDA_SAFE_CALL(cudaMalloc((void **)&hitLists, sizeof(int)*t.leafCount));
    CUDA_SAFE_CALL(cudaMemset(hitLists, -1, sizeof(int)*t.leafCount));

    int roundedNPositions = ((nPositions-1)/INCLUDE_CHUNK_SIZE+1)*INCLUDE_CHUNK_SIZE;

    printTree<<<1, 1>>>(tree);

    printf("Scattering nodes into tree\n"); fflush(stdout);
    // scatter the position into the tree
    _includeLookup<<<roundedNPositions/INCLUDE_CHUNK_SIZE, INCLUDE_CHUNK_SIZE>>>(tree, devicePositions, nPositions, includeHits, hitLists);


    int roundedNLeaves = ((t.leafCount-1)/INCLUDE_CHUNK_SIZE+1)*INCLUDE_CHUNK_SIZE;
    
    printf("Growing new leaves\n"); fflush(stdout);
    // make any new nodes necessary
    _reduceIncludeHitLists<<<roundedNLeaves/INCLUDE_CHUNK_SIZE, INCLUDE_CHUNK_SIZE>>>(tree, t.leafCount, devicePositions, includeHits, hitLists);

    CUDA_SAFE_CALL(cudaMemcpy(&t, tree, sizeof(kd_tree), cudaMemcpyDeviceToHost));    
    printf("Tree now has %i leaves\n", t.leafCount);

    printTree<<<1, 1>>>(tree);

    // clean up
    CUDA_SAFE_CALL(cudaFree(devicePositions));
    CUDA_SAFE_CALL(cudaFree(includeHits));
    CUDA_SAFE_CALL(cudaFree(hitLists));
}

static __global__ void _includeLookup(kd_tree *tree, float *positions, int nPositions,
				      include_hit *includeHits, int *hitLists) {

    int dimensions = tree->positionDimensions;
    const int myIdx = blockIdx.x * blockDim.x + threadIdx.x;
    float *queryBlock = positions + dimensions * blockIdx.x * INCLUDE_CHUNK_SIZE;

    // put my vector in shared memory
    __shared__ float localQuery[INCLUDE_CHUNK_SIZE*MAX_DIMENSIONS];
    for (int i = threadIdx.x; i < dimensions*INCLUDE_CHUNK_SIZE; i += INCLUDE_CHUNK_SIZE) {
	localQuery[i] = queryBlock[i];
    }

    float *myQuery = localQuery + threadIdx.x * dimensions;

    __syncthreads();

    if (myIdx >= nPositions) return;

    // ok, we have my query, we're good to go
    int nodeIdx = 0;
    node *nodeArray = tree->nodeArray;
    int parent = -1;
    DEBUG(printf("Query %i:\n", myIdx));
    do {
	node n = nodeArray[nodeIdx];	
	if (myQuery[n.cut_dim] <= n.cut_val) {
	    DEBUG(printf(" left (%f vs %f) to node %i\n", myQuery[n.cut_dim], n.cut_val, n.left));
	    parent = -nodeIdx;
	    nodeIdx = n.left;
	} else {
	    DEBUG(printf(" right (%f vs %f) to node %i\n", myQuery[n.cut_dim],  n.cut_val, n.right));
	    parent = nodeIdx;
	    nodeIdx = n.right;	
	}
    } while (nodeIdx > 0);

    // we're at a leafnode
    nodeIdx *= -1;

    // check if I'm catered for by this leaf
    float distance = 0;
    float *leaf = tree->leafPositions + nodeIdx*dimensions;
    for (int i = 0; i < dimensions; i++) {
	float d = leaf[i] - myQuery[i];
	distance += d*d;
    }

    // if not, add myself as a hit to this leaf
    if (distance > BUILD_SIZE_LIMIT*BUILD_SIZE_LIMIT) {
	// set my hit record, and push it on the front of the appropriate leaf list
	// the next person in the list is whomever I clobbered
	includeHits[myIdx].next = atomicExch(hitLists + nodeIdx, myIdx);
	includeHits[myIdx].parent = parent;
    }
}

// either builds a new split node and sets the owner as appropriate, or decides it isn't necessary and doesn't.
static __device__ int _growFromLeaf(kd_tree *tree, int parent, int positionDimensions,
				    int existingLeafIdx, float *proposedLeaf) {
    // find the dimension of maximum distance, and also the total
    // distance
    float distance = 0, length = -1;
    int longest = 0;
    bool newLeafGoesOnRight = false;

    DEBUG(printf(" considering adding a new leaf node at leaf index %i\n", existingLeafIdx); fflush(stdout));

    float *existingLeaf = tree->leafPositions + existingLeafIdx * positionDimensions;

    for (int i = 0; i < positionDimensions; i++) {
	float delta = existingLeaf[i] - proposedLeaf[i];
	bool negative = delta < 0;
	float d = negative ? -delta : delta;
	if (d > length) {
	    longest = i;
	    length = d;
	    newLeafGoesOnRight = negative;
	}
	distance += delta*delta;
    }

    // check if there's even a need to put a new leaf node here
    if (distance < 1) return -1;

    DEBUG(printf("  yes, distance is %f\n", distance); fflush(stdout));

    // there is, so split
    int newNodeIdx = atomicAdd(&tree->nodeCount, 1);
    int newLeafIdx = atomicAdd(&tree->leafCount, 1);

    DEBUG(printf("  new node and leaf indices: %i %i\n", newNodeIdx, newLeafIdx); fflush(stdout));

    float *newLeaf = tree->leafPositions + newLeafIdx*positionDimensions;
    for (int i = 0; i < positionDimensions; i++) {
	newLeaf[i] = proposedLeaf[i];
    }
    
    float split = (proposedLeaf[longest] + existingLeaf[longest])/2;

    DEBUG(printf("  splitting on dimension %i at value %f\n", longest, split));

    node *n = tree->nodeArray + newNodeIdx;
    n->cut_val = split;
    n->cut_dim = longest;
    n->parent = (parent > 0 ? parent : -parent);
    if (newLeafGoesOnRight) {
	n->left = -existingLeafIdx;
	n->right = -newLeafIdx;
    } else {
	n->left = -newLeafIdx;
	n->right = -existingLeafIdx;
    }

    if (parent > 0) {
	tree->nodeArray[parent].right = newNodeIdx;
    } else {
	tree->nodeArray[-parent].left = newNodeIdx;
    }

    return newNodeIdx;
}

static __device__ void _growFromInnerNode(kd_tree *tree, int nextNodeIdx, int positionDimensions, float *newPosition) {
    int parent = nextNodeIdx;
    node *nodeArray = tree->nodeArray;
    while (nextNodeIdx > 0) {
	DEBUG(printf("Retrieving node %i\n", nextNodeIdx); fflush(stdout));

	node *n = nodeArray + nextNodeIdx;

	if (newPosition[n->cut_dim] > n->cut_val) {
	    parent = nextNodeIdx;
	    nextNodeIdx = n->right;
	} else {
	    parent = -nextNodeIdx;
	    nextNodeIdx = n->left;
	}
    }
    
    _growFromLeaf(tree, parent, positionDimensions, -nextNodeIdx, newPosition);
}

static __global__ void _reduceIncludeHitLists(kd_tree *tree, int oldLeafCount, float *positions, include_hit *hits, int *hitLists) {

    int positionDimensions = tree->positionDimensions;

    const int leafIdx = blockIdx.x*INCLUDE_CHUNK_SIZE + threadIdx.x;
    if (leafIdx >= oldLeafCount) return;

    DEBUG(printf("Growing at leaf %i\n", leafIdx));

    // follow the result list, gathering all the hits
    int nextHitIdx = hitLists[leafIdx];

    if (nextHitIdx < 0) return;

    DEBUG(printf(" got at least one node (%i)\n", nextHitIdx); fflush(stdout));
    
    // grab the first hit
    include_hit hit = hits[nextHitIdx];

    // make a new internal node for it
    int newRoot =  _growFromLeaf(tree, hit.parent, positionDimensions,
				 leafIdx, positions + nextHitIdx*positionDimensions);

    if (newRoot < 0) newRoot = hit.parent;
    if (newRoot < 0) newRoot = -newRoot;

    nextHitIdx = hit.next;

    // deal with the other hits
    while (nextHitIdx >= 0) {	
	DEBUG(printf(" got another node (%i), parent is %i\n", nextHitIdx, newRoot); fflush(stdout););
	hit = hits[nextHitIdx];

	_growFromInnerNode(tree, newRoot, positionDimensions, positions + nextHitIdx*positionDimensions);

	nextHitIdx = hit.next;
    }
}
