
#define LOOKUP_CHUNK_SIZE 16
#define JOBS_PER_THREAD 4

typedef enum {EMPTY = 0, READY, LOCKED} lookup_job_state;

struct lookup_job {
    // the state of this job slot (empty, ready, or locked)
    int state;
    // which (local) query is here, and how many samples does it represent
    unsigned short queryIdx, samples;
    // what is the probability with which one sample from this query made it this far
    float probability;
    // what node is this sample at
    int nodeIdx;
};


__device__ inline void atomicFloatAdd(float *address, float val) {
      int i_val = __float_as_int(val);
      int tmp0 = 0;
      int tmp1;

      while( (tmp1 = atomicCAS((int *)address, tmp0, i_val)) != tmp0)
      {
              tmp0 = tmp1;
              i_val = __float_as_int(val + __int_as_float(tmp1));
      }
}

typedef enum {SLICE=0, SPLAT} query_type;

__device__ void handleLeaf(const kd_tree &tree, int leafIdx, float *myQuery,
			   float *myValue, float invSigma, float w, query_type type) {

    if (w == 0) return;
    if (isinf(w)) {
	DEBUG(printf("%i) Infinite weight!!\n", threadIdx.x));
    }
    // figure out the weight
    float *leafPos = tree.leafPositions + leafIdx*tree.positionDimensions;    
    float distance = 0;
    for (int i = 0; i < tree.positionDimensions; i++) {
	float d = myQuery[i] - leafPos[i];
	distance += d*d;
    }
    float weight = __expf(-distance*0.5*invSigma*invSigma); 
    // - 0.006738;
	    
    if (weight > 0.006738) {
	weight *= w;

	float *leafVal = tree.leafValues + leafIdx * tree.valueDimensions;
	// add in the value I found
	if (type == SLICE) {
	    for (int i = 0; i < tree.valueDimensions; i++) {
		atomicFloatAdd(myValue + i, weight*leafVal[i]);
	    }           
	} else if (type == SPLAT) {
	    for (int i = 0; i < tree.valueDimensions; i++) {
		atomicFloatAdd(leafVal + i, weight*myValue[i]);
	    }
	}
    }   
}

__device__ int reserveSlot(lookup_job *lookupJobs) {
    int i = threadIdx.x * JOBS_PER_THREAD;
    while (1) {
	if (lookupJobs[i].state == EMPTY) {
	    int claim = atomicExch(&lookupJobs[i].state, LOCKED);
	    if (claim == EMPTY) {
		return i;
	    } else if (claim == READY) {
		// oops, this slot was filled
		atomicExch(&lookupJobs[i].state, READY);
	    }
	}	       
		    
	i++;
	if (i == JOBS_PER_THREAD*LOOKUP_CHUNK_SIZE) i = 0;	    
	if (i == threadIdx.x*JOBS_PER_THREAD) return -1;
    }
}

__global__ void _gaussianLookup(kd_tree tree, float *positions, float *values, 
				int nPositions, float sigma, int samples, uint4 seed, query_type type) {    
    const int myIdx = blockIdx.x * blockDim.x + threadIdx.x;
    positions += tree.positionDimensions * blockIdx.x * LOOKUP_CHUNK_SIZE;
    if (type == SLICE) {
	values += tree.valueDimensions * blockIdx.x * LOOKUP_CHUNK_SIZE;
    } else {
	values += (tree.valueDimensions-1) * blockIdx.x * LOOKUP_CHUNK_SIZE;
    }

    // make my seed unique to me
    seed.x ^= myIdx;
    seed.y ^= myIdx;
    seed.z ^= myIdx;
    seed.w ^= myIdx;

    // put the positions and values in shared memory
    __shared__ float localQuery[LOOKUP_CHUNK_SIZE*MAX_DIMENSIONS];
    for (int i = threadIdx.x; i < tree.positionDimensions*LOOKUP_CHUNK_SIZE; i += LOOKUP_CHUNK_SIZE) {
	localQuery[i] = positions[i];
    }
    
    __shared__ float localValue[LOOKUP_CHUNK_SIZE*MAX_VALUE_DIMENSIONS];


    // make the work queue
    __shared__ lookup_job lookupJobs[LOOKUP_CHUNK_SIZE*JOBS_PER_THREAD];
    for (int i = 0; i < JOBS_PER_THREAD; i++) {
	lookupJobs[threadIdx.x*JOBS_PER_THREAD + i].state = EMPTY;
    }

    __shared__ int jobsInProgress, jobsInQueue;
    if (threadIdx.x == 0) jobsInProgress = jobsInQueue = 0;
    __syncthreads();

    lookup_job currentJob;
    if (myIdx < nPositions) {
	currentJob.queryIdx = threadIdx.x;
	currentJob.samples = samples;
	currentJob.probability = 1;
	currentJob.state = READY;
	currentJob.nodeIdx = 0;
	atomicAdd(&jobsInProgress, 1);

	// if we're slicing, no need to retrieve the old values, as we clobber them
	// if we're splatting, we want them, as we use them
	float *myLocalValue = localValue + threadIdx.x * tree.valueDimensions;
	if (type == SLICE) {
	    for (int i = 0; i < tree.valueDimensions; i++) {
		myLocalValue[i] = 0;
	    }
	} else {
	    float *myValue = values + threadIdx.x * (tree.valueDimensions-1);
	    for (int i = 0; i < tree.valueDimensions-1; i++) {
		myLocalValue[i] = myValue[i];
	    }
	    myLocalValue[tree.valueDimensions-1] = 1;
	}

    } else {
	currentJob.state = EMPTY;	
    }

    float invSigma = 1.0f/sigma;  

    while (1) {
	DEBUG(__syncthreads());
	DEBUG(printf("%i) jobs in progress: %i jobs in queue: %i\n", threadIdx.x, jobsInProgress, jobsInQueue));
	if (jobsInProgress == 0 && jobsInQueue == 0) break;
	DEBUG(__syncthreads());

	// STAGE 1: look for work
	if (currentJob.state != READY) {
	    DEBUG(printf("%i) looking for work\n", threadIdx.x));

	    int j = threadIdx.x*JOBS_PER_THREAD;
	    // try to steal work from neighbours up to 3 ahead
	    for (int k = 0; k < JOBS_PER_THREAD*4; k++) {
		//DEBUG(printf("%i) looking in slot %i for a job\n", threadIdx.x, j));
		if (lookupJobs[j].state == READY) {
		    int claim = atomicExch(&lookupJobs[j].state, LOCKED);
		    if (claim == READY) goto found_work;
		    if (claim == EMPTY) {
			// oops, I just locked an empty slot
			atomicExch(&lookupJobs[j].state, EMPTY);
		    }
		    // if claim is locked, no point reverting
		}
		// no luck with this slot, move on to the next
		j++;
		if (j == JOBS_PER_THREAD*LOOKUP_CHUNK_SIZE) j = 0;	    
		if (j == threadIdx.x*JOBS_PER_THREAD) break;
	    }
	    
	    // I didn't find anything to do!
	    continue;

	  found_work:
	    DEBUG(printf("%i) Popping job %i off the queue\n", threadIdx.x, j)); 
	    atomicAdd(&jobsInProgress, 1);
	    currentJob = lookupJobs[j];
	    currentJob.state = READY;
	    atomicAdd(&jobsInQueue, -1);
	    lookupJobs[j].state = EMPTY;
	} else {
	    DEBUG(printf("%i) continuing old job at node %i with %i samples\n", threadIdx.x, currentJob.nodeIdx, currentJob.samples));
	}

	DEBUG(printf("%i) job:\n"
		     " queryIdx = %i\n"
		     " nodeIdx  = %i\n"
		     " samples  = %i\n"
		     " prob     = %f\n"
		     " state    = %i\n",
		     threadIdx.x,
		     currentJob.queryIdx,
		     currentJob.nodeIdx,
		     currentJob.samples,
		     currentJob.probability,
		     currentJob.state));

	// STAGE 2: walk this node down until it's done, or it diverges
	float *myQuery = localQuery + currentJob.queryIdx * tree.positionDimensions;
	float *myValue = localValue + currentJob.queryIdx * tree.valueDimensions;
	int samplesLeft, samplesRight;
	int leftNodeIdx, rightNodeIdx;
	float pLeft;
	do {
	    node n = tree.nodeArray[currentJob.nodeIdx];
	    leftNodeIdx = n.left;
	    rightNodeIdx = n.right;
	    float pMin = gaussianApproxCDF((n.min_val - myQuery[n.cut_dim]) * invSigma);
	    float pMax = gaussianApproxCDF((n.max_val - myQuery[n.cut_dim]) * invSigma);
	    pLeft = gaussianApproxCDF((n.cut_val - myQuery[n.cut_dim]) * invSigma);	    
	    float samplesLeftF = pLeft * currentJob.samples;
	    // send a bunch left and right
	    samplesLeft = samplesLeftF;
	    samplesRight = currentJob.samples - samplesLeftF;
	    DEBUG(printf("pLeft = %f\n"
			 "samplesLeftF = %f\n"
			 "samplesLeft = %i\n"
			 "samples = %i\n"
			 "samplesRight = %i\n",
			 pLeft, samplesLeftF, samplesLeft, currentJob.samples, samplesRight));

	    // flip a coin for the last one
	    if (samplesLeft + samplesRight < currentJob.samples) {
		samplesLeftF -= samplesLeft;
		if (samplesLeftF > 0.999 || samplesLeftF > 0.001 && urand(&seed) < samplesLeftF) {
		    samplesLeft++;
		} else {
		    samplesRight++;
		}
	    }
	    DEBUG(printf("%i) Value %f at split %f %f %f. Split is %i %i\n", threadIdx.x,
			 myQuery[n.cut_dim], n.min_val, n.cut_val, n.max_val, samplesLeft, samplesRight));

	    if (samplesLeft == 0) {
		currentJob.nodeIdx = n.right;
		currentJob.probability *= (1-pLeft);
		DEBUG(if (currentJob.probability < 0.000001) printf("ERROR A: %f %f\n", currentJob.probability, pLeft));
	    } else if (samplesRight == 0) {
		currentJob.nodeIdx = n.left;
		currentJob.probability *= pLeft;
		DEBUG(if (currentJob.probability < 0.000001) printf("ERROR B: %f %f\n", currentJob.probability, pLeft));
	    } else break;
	} while (currentJob.nodeIdx > 0);

	// at this point we're either at a leaf, or have diverged
	if (currentJob.nodeIdx <= 0 && (samplesLeft == 0 || samplesRight == 0)) {
	    DEBUG(printf("%i) At a leaf %i\n", threadIdx.x, currentJob.nodeIdx));
	    handleLeaf(tree, -currentJob.nodeIdx, myQuery, myValue,
		       invSigma, currentJob.samples/currentJob.probability, type);
	    currentJob.state = EMPTY;
	    atomicAdd(&jobsInProgress, -1);
	} else {
	    // we've diverged, if one or both children is a leaf we should just retire them now
	    if (leftNodeIdx <= 0 && rightNodeIdx <= 0) {
		handleLeaf(tree, -leftNodeIdx, myQuery, myValue, invSigma,
			   samplesLeft/(pLeft*currentJob.probability), type);
		handleLeaf(tree, -rightNodeIdx, myQuery, myValue, invSigma,
			   samplesRight/((1-pLeft)*currentJob.probability), type);
		atomicAdd(&jobsInProgress, -1);
		currentJob.state = EMPTY;
		continue;
	    } 
	    if (leftNodeIdx <= 0) {
		handleLeaf(tree, -leftNodeIdx, myQuery, myValue, invSigma,
			   samplesLeft/(pLeft*currentJob.probability), type);
		currentJob.nodeIdx = rightNodeIdx;
		currentJob.probability *= (1-pLeft);
		DEBUG(if (currentJob.probability < 0.000001) printf("ERROR C: %f %f\n", currentJob.probability, pLeft));
		currentJob.samples = samplesRight;
		continue;
	    }
	    if (rightNodeIdx <= 0) {
		handleLeaf(tree, -rightNodeIdx, myQuery, myValue, invSigma,
			   samplesRight/((1-pLeft)*currentJob.probability), type);
		currentJob.nodeIdx = leftNodeIdx;
		currentJob.probability *= pLeft;
		DEBUG(if (currentJob.probability < 0.000001) printf("ERROR D: %f %f\n", currentJob.probability, pLeft));
		currentJob.samples = samplesLeft;
		continue;
	    }
	    
	    // try to reserve a slot to defer one of these jobs until later
	    int slotReserved = (jobsInQueue == JOBS_PER_THREAD*LOOKUP_CHUNK_SIZE) ? -1 : reserveSlot(lookupJobs);
	    
	    if (slotReserved >= 0) {
		DEBUG(printf("%i) Successfully reserved output slot %i for half of this job\n", threadIdx.x, slotReserved));
		// put the larger of the two in the reserved output slot,
		// and keep working on the smaller
		if (samplesLeft > samplesRight) {
		    lookupJobs[slotReserved].nodeIdx = leftNodeIdx;
		    lookupJobs[slotReserved].probability = currentJob.probability*pLeft;
		    DEBUG(if (lookupJobs[slotReserved].probability < 0.000001) printf("ERROR E: %f %f\n", currentJob.probability, pLeft));
		    lookupJobs[slotReserved].samples = samplesLeft;
		    lookupJobs[slotReserved].queryIdx = currentJob.queryIdx;
		    lookupJobs[slotReserved].state = READY;
		    atomicAdd(&jobsInQueue, 1);
		    currentJob.nodeIdx = rightNodeIdx;
		    currentJob.probability *= (1-pLeft);
		    currentJob.samples = samplesRight;
		} else {
		    lookupJobs[slotReserved].nodeIdx = rightNodeIdx;
		    lookupJobs[slotReserved].probability = currentJob.probability*(1-pLeft);
		    DEBUG(if (lookupJobs[slotReserved].probability < 0.000001) printf("ERROR F: %f %f\n", currentJob.probability, pLeft));
		    lookupJobs[slotReserved].samples = samplesRight;
		    lookupJobs[slotReserved].queryIdx = currentJob.queryIdx;
		    lookupJobs[slotReserved].state = READY;
		    atomicAdd(&jobsInQueue, 1);
		    currentJob.nodeIdx = leftNodeIdx;
		    currentJob.probability *= pLeft;
		    currentJob.samples = samplesLeft;
		}
		
	    } else {
		// iterate over samples instead
		DEBUG(printf("%i) Iterating over %i samples\n", threadIdx.x, currentJob.samples));
		for (int i = 0; i < currentJob.samples; i++) {
		    int nodeIdx = currentJob.nodeIdx;
		    float p = currentJob.probability;
		    do {
			node n = tree.nodeArray[nodeIdx];
			float pMin = gaussianApproxCDF((n.min_val - myQuery[n.cut_dim]) * invSigma);
			float pMax = gaussianApproxCDF((n.max_val - myQuery[n.cut_dim]) * invSigma);
			pLeft = gaussianApproxCDF((n.cut_val - myQuery[n.cut_dim]) * invSigma);	    

			if (pLeft > 0.999 || pLeft > 0.001 && urand(&seed) < pLeft) {
			    nodeIdx = n.left;
			    p *= pLeft;
			} else {
			    nodeIdx = n.right;
			    p *= (1-pLeft);
			}
		    } while (nodeIdx > 0);

		    handleLeaf(tree, -nodeIdx, myQuery, 
			       myValue, invSigma, 1.0f/p, type);
		}

		// retire this job
		currentJob.state = EMPTY;
		atomicAdd(&jobsInProgress, -1);
	    }
	}	
    }
    
    // we're all done

    // wait for everyone to agree we finished
    __syncthreads();

    // write back all the data
    if (type == SLICE) {
	DEBUG(printf("%i out of loop, writing data\n", threadIdx.x));
	for (int i = threadIdx.x; i < tree.valueDimensions*LOOKUP_CHUNK_SIZE; i += LOOKUP_CHUNK_SIZE) {
	    values[i] = localValue[i];
	}
    }

    DEBUG(printf("%i is done\n", threadIdx.x); fflush(stdout));
}
