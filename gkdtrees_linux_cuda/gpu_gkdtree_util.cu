#define MODULUS  ((double) 4294967296.0) // 2^32 as a double
#define MODULUS_INV ((float) (1.0 / MODULUS))  

// a uniform random float between zero and one
__device__ float urand(uint4 *seed) {
    // George Marsaglia's KISS generator from his post to
    // to sci.stat.math on 1/12/99    

    // a pair of MWC's
    seed->x = 36969*(seed->x & 0xffff) + (seed->x >> 16);
    seed->y = 18000*(seed->y & 0xffff) + (seed->y >> 16);   
    unsigned int z = (seed->x << 16) + seed->y;

    // a shift register
    seed->z ^= (seed->z << 17);
    seed->z ^= (seed->z >> 13);
    seed->z ^= (seed->z << 5);

    // a linear congruential generator
    seed->w = 69069*seed->w + 1234567;

    z ^= seed->w;
    z += seed->z;
    
    return z * MODULUS_INV;
}

__global__ void urandTest() {
    #ifdef EMULATION
    uint4 seed = make_uint4(12345, 56789, 123, 4325);
    for (int i = 0; i < 10000; i++) {
	printf("%f\n", urand(&seed));
    }
    #endif
}

// draw a random sample from an distribution that approximates a gaussian of std dev 1
__device__ float gaussianApproxRand(uint4 *seed) {
    return (urand(seed) + urand(seed) + urand(seed))*2 - 3;
}

// the PDF of the aforementioned distribution
__device__ float gaussianApproxPDF(float x) {
    if (x < -3) {
	return 0;
    }
    if (x < -1) {
	x += 3;
	return x*x/16;
    }
    if (x < 1) {
	x = 3-x;
	return x*x/8;
    }
    if (x < 3) {
	x -= 3;
	return x*x/16;
    }
    return 0;
}

// the CDF of the aforementioned distribution
__device__ float gaussianApproxCDF(float x) {
    if (x < -3) {
	return 0;
    }
    if (x < -1) {
	x += 3;
	return x*x*x/48;
    }
    if (x < 1) {
	return (12 + x*(9 - x*x))/24;
    }
    if (x < 3) {
	x -= 3;
	return 1 + x*x*x/48;
    }
    return 1;
}



// A function that prints the whole tree. Only useful in emulation mode.

#ifdef EMULATION
__device__ void printNode(kd_tree *t, int nodeIdx, int depth) {
    char space[256];    
    for (int i = 0; i < depth; i++) space[i] = '.';
    space[depth] = 0; 
    if (nodeIdx > 0 || depth == 0) {
	node n = t->nodeArray[nodeIdx];
	printf("%sNode %i: %i %f (%f %f) (%i %i %i)\n",
	       space, nodeIdx,
	       n.cut_dim, n.cut_val, n.min_val,
	       n.max_val, n.parent, n.left, n.right);
	printNode(t, n.left, depth+1);
	printNode(t, n.right, depth+1);
    } else {
	printf("%sLeaf %i: \n", space, -nodeIdx);

	printf("%s Position: ", space);
	for (int i = 0; i < t->positionDimensions; i++) {
	    printf("%3.3f ", t->leafPositions[-nodeIdx*t->positionDimensions+i]);
	}
	printf("\n%s Value: ", space);
	for (int i = 0; i < t->valueDimensions; i++) {
	    printf("%3.3f ", t->leafValues[-nodeIdx*t->valueDimensions+i]);
	}
	printf("\n");
    }
}
#endif

__global__ void printTree(kd_tree *t) {
#ifdef EMULATION
    printf("tree:\n"
	   " nodeArray          = %x\n"
	   " leafPositions      = %x\n"
	   " leafValues         = %x\n"
	   " nodeCount          = %i\n"
	   " leafCount          = %i\n"
	   " positionDimensions = %i\n"
	   " valueDimensions    = %i\n",
	   t->nodeArray, t->leafPositions, t->leafValues,
	   t->nodeCount, t->leafCount, t->positionDimensions,
	   t->valueDimensions);
    printNode(t, 0, 0);
    printf("End of tree\n");
    fflush(stdout);
#endif
}


