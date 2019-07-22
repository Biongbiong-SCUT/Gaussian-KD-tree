
static __global__ void _finalizeTree(kd_tree tree);

#define FINALIZE_CHUNK_SIZE 256

void finalizeTree() {
    // grab the tree header
    kd_tree t;
    CUDA_SAFE_CALL(cudaMemcpy(&t, tree, sizeof(kd_tree), cudaMemcpyDeviceToHost));

    int roundedNodeCount = ((t.nodeCount - 1) / FINALIZE_CHUNK_SIZE + 1);

    _finalizeTree<<<roundedNodeCount, FINALIZE_CHUNK_SIZE>>>(t);

}

static __global__ void _finalizeTree(kd_tree tree) {
    int myNodeIdx = threadIdx.x + blockDim.x * blockIdx.x;

    if (myNodeIdx >= tree.nodeCount) return;

    node myNode = tree.nodeArray[myNodeIdx];
    int parent = myNode.parent;

    myNode.max_val = INF;
    myNode.min_val = -INF;
 
    // now we walk up the tree, figuring out the bounds in the cut dimension
    DEBUG(printf("At node %i, walking upwards\n", myNodeIdx));
    while (parent >= 0) {
	DEBUG(printf(" retrieving node %i\n", parent));
	node currentNode = tree.nodeArray[parent];
	if (currentNode.cut_dim == myNode.cut_dim) {
	    if (myNode.cut_val < currentNode.cut_val) {
		myNode.max_val = fminf(myNode.max_val, currentNode.cut_val);
	    } else {
		myNode.min_val = fmaxf(myNode.min_val, currentNode.cut_val);
	    }
	}
	parent = currentNode.parent;
    }

    // now write the node back
    tree.nodeArray[myNodeIdx].min_val = myNode.min_val;
    tree.nodeArray[myNodeIdx].max_val = myNode.max_val;
}




