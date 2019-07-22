
#include <ImageStack.h>
#include "eigenvectors.h"
#include "device.h"

using namespace ImageStack;

struct header {
    int frames, width, height, channels;
};

class VolumePatchStream  {
public:
    VolumePatchStream(char *filename, int patchFrames_, int patchWidth_, int patchHeight_) {
	// peek in the top of the tmp file
	file = fopen(filename, "rb");

	patchFrames = patchFrames_;
	patchWidth = patchWidth_;
	patchHeight = patchHeight_;

	slices = new Image *[patchFrames];

	header h;
	fread(&h, sizeof(header), 1, file);
	
	// Allocate the in-core memory
	frames = h.frames;
	for (int i = 0; i < patchFrames; i++) {
	    slices[i] = new Image(1, h.width, h.height, h.channels);
	}
	
	// make the mask
	mask = new float[patchWidth*patchHeight*patchFrames];
	for (int i = 0; i < patchFrames; i++) {
	    for (int j = 0; j < patchHeight; j++) {
		for (int k = 0; k < patchWidth; k++) {
		    float di = (i - patchFrames/2)*4;
		    float dj = (j - patchHeight/2)*4;
		    float dk = (k - patchWidth/2)*4;
		    di /= patchFrames;
		    dj /= patchHeight;
		    dk /= patchWidth;
		    mask[(i*patchHeight+j)*patchWidth+k] = expf(-(di*di + dj*dj + dk*dk));
		}
	    }
	}

	// allocate space for the patch
	patch = new float[patchWidth*patchHeight*patchFrames*h.channels];

	t = 1;
	reset();
    }


    float *getMask() {
	return mask;
    }

    float *next() {

	if (t >= frames - (patchFrames-1)) return NULL;
	
	int channels = slices[0]->channels;
	float *mPtr = mask;
	float *pPtr = patch;
	float *iPtr;
	if (channels > 1) {
	    for (int dt = 0; dt < patchFrames; dt++) {
		for (int dy = 0; dy < patchWidth; dy++) {
		    iPtr = (*(slices[dt]))(x, y+dy);
		    for (int dx = 0; dx < patchHeight; dx++) {
			float m = *mPtr++;
			for (int c = 0; c < channels; c++) {		    
			    *pPtr++ = *iPtr++ * m;
			}
		    }
		}
	    }
	} else {
	    for (int dt = 0; dt < patchFrames; dt++) {
		for (int dy = 0; dy < patchWidth; dy++) {
		    iPtr = (*(slices[dt]))(x, y+dy);
		    for (int dx = 0; dx < patchHeight; dx++) {
			*pPtr++ = (*iPtr++)*(*mPtr++);
		    }
		}
	    }
	}

	skip();

	return patch;

    }
    
    void skip() {
	if (t >= frames - (patchFrames-1)) return;

	x++;
	if (x == slices[0]->width-(patchWidth-1)) {
	    // done with this row
	    x = 0;
	    y++;
	}

	if (y == slices[0]->height - (patchHeight-1)) {
	    // done with this slice
	    x = y = 0;
	    t++;

	    if (t < frames - (patchFrames-1)) {		
		printf("Loading slice %i\n", t);
		
		// load the next slice
		fread((*(slices[0]))(0, 0), sizeof(float), slices[0]->width * slices[0]->height * slices[0]->channels, file);
		
		// rotate the pointers
		Image *tmp = slices[0];
		for (int i = 0; i < patchFrames-1; i++) {
		    slices[i] = slices[i+1];
		}
		slices[patchFrames-1] = tmp;
	    }
	}
    }

    void reset() {
	x = y = 0;	

	if (t != 0) {
	    // fseek to just past the header
	    fseek(file, 4*sizeof(int), SEEK_SET);
	    
	    // load the first PATCH frames
	    for (int i = 0; i < patchFrames; i++) {
		fread((*(slices[i]))(0, 0), sizeof(float), slices[0]->width * slices[0]->height * slices[0]->channels, file);
	    }
	}
	
	t = 0;
    }

    int channels() {
	return slices[0]->channels*patchFrames*patchWidth*patchHeight;
    }

    int size() {
	return (frames-patchFrames+1) * (slices[0]->width-patchWidth+1) * (slices[0]->height-patchHeight+1);
    }

private:
    int patchFrames, patchWidth, patchHeight;
    Image **slices;
    float spatialScale;
    FILE *file;
    int frames;
    int x, y, t;
    float *patch, *mask;
};


int main(int argc, char **argv) {
    if (argc != 7) {
	printf("Usage: patchPCA.exe input.tmp output.tmp <output dimensions> <patch frames> <patch width> <patch height>\n");
	return 1;
    }

    initCuda(1, argv);

    char *inputFilename = argv[1];
    char *outputFilename = argv[2];
    int dimensions = atoi(argv[3]);
    int patchFrames = atoi(argv[4]);
    int patchWidth = atoi(argv[5]);
    int patchHeight = atoi(argv[6]);

    // Load a random subsample of the input, compute a covariance
    // matrix and mean on it, and find the strongest eigenvectors
    header h;    
    FILE *f = fopen(inputFilename, "rb");
    fread(&h, sizeof(header), 1, f);
    fclose(f);
    int patchSize = patchWidth*patchHeight*patchFrames*h.channels;

    VolumePatchStream stream(inputFilename, patchFrames, patchWidth, patchHeight);
    int desired = (1<<16); // 64000
    int remaining = stream.size();
    if (desired > remaining) desired = remaining;
    Eigenvectors eig(patchSize, dimensions);

    printf("Take %i out of %i elements to compute covariance.\n", desired, remaining);
    while (desired && remaining) {
	double r = (double)rand()/RAND_MAX;
	if (r*remaining < desired) {
	    desired--;
	    float *p = stream.next();
	    if (p) eig.add(p);
	} else {
	    stream.skip();
	}
	remaining--;		
    }

    printf("Computing covariance\n");
   
    eig.compute();

    // for each block of the input, convolve by the eigenvectors, and save the block out
    int xBlocks = (h.width-1)/(2048*h.channels) + 1;
    int yBlocks = (h.height-1)/2048 + 1;
    int tBlocks = (h.frames-1)/2048 + 1;
    int blockWidth, blockFrames, blockHeight;

  retry:
    printf("Testing blocking scheme %i x %i x %i\n", tBlocks, xBlocks, yBlocks);
    // figure out how much memory a single block would take under
    // the current scheme
    blockWidth = (h.width - 1) / xBlocks + patchWidth;
    
    if (blockWidth * h.channels > 2048) {
	xBlocks*=2;
	goto retry;
    }

    blockHeight = (h.height - 1) / yBlocks + patchHeight;
    
    if (blockHeight > 2048) {
	yBlocks*=2;
	goto retry;
    }
    
    blockFrames = (h.frames - 1) / tBlocks + patchFrames;
    
    if (blockFrames > 2048) {
	tBlocks*=2;
	goto retry;
    }
    
    int memory = blockWidth*blockHeight*blockFrames*dimensions*sizeof(float);
    if (memory > (1 << 26)) { // 64MB
	// preferentially split on dimensions the filter doesn't span
	if (patchFrames == 1) {
	    tBlocks *= 2;
	} else if (patchWidth == 1) {
	    xBlocks *= 2;
	} else if (patchHeight == 1) {
	    yBlocks *= 2;
	} else if (blockFrames > blockWidth && blockFrames > blockHeight) {
	    tBlocks *= 2;
	} else if (blockWidth > blockHeight) {
	    xBlocks *= 2;
	} else {
	    yBlocks *= 2;
	}
	goto retry;
    }

    printf("Subdividing the image into %i x %i x %i blocks for convolution\n", tBlocks, xBlocks, yBlocks);
    printf("Each block is %i x %i x %i\n", blockFrames, blockWidth, blockHeight);

    // Load each block
    int dt = blockFrames - patchFrames+1;
    int dx = blockWidth - patchWidth+1;
    int dy = blockHeight - patchHeight+1;

    Image out(blockFrames, blockWidth, blockHeight, dimensions);

    float *kernel = new float[patchFrames*patchWidth*patchHeight*dimensions*h.channels];
    float *mask = stream.getMask();
    for (int d = 0; d < dimensions; d++) {
	double *v = eig.getEigenvector(d);
	for (int i = 0; i < patchFrames*patchWidth*patchHeight; i++) {
	    for (int j = 0; j < h.channels; j++) {
		kernel[(i*dimensions + d)*h.channels + j] = v[i*h.channels + j] * mask[i];
	    }
	}
    }

    for (int d = 0; d < dimensions; d++) {
	char filename[128];
	sprintf(filename, "kernel_%02i.tmp", d);
	FILE *f = fopen(filename, "wb");
	int header[] = {patchFrames, patchWidth, patchHeight, h.channels};
	fwrite(header, 4, sizeof(int), f);
	for (int i = 0; i < patchFrames*patchWidth*patchHeight; i++) {
	    for (int c = 0; c < h.channels; c++) {
		fwrite(kernel + (i*dimensions + d)*h.channels + c, 1, sizeof(float), f);
	    }
	}
	fclose(f);
    }

    // make the output
    CreateTmp::apply(outputFilename, h.frames, h.width, h.height, dimensions);

    for (int t = 0; t < tBlocks; t++) {
	for (int y = 0; y < yBlocks; y++) {
	    for (int x = 0; x < xBlocks; x++) {
		printf("Loading block from %i x %i x %i of size %i x %i x %i\n",
		       t*dt-patchFrames/2, x*dx-patchWidth/2, y*dy-patchHeight/2,
		       blockFrames, blockWidth, blockHeight);		       
		fflush(stdout);
		Image block = LoadBlock::apply(string(inputFilename), 
					       t*dt-patchFrames/2, x*dx-patchWidth/2, y*dy-patchHeight/2, 0,
					       blockFrames, blockWidth, blockHeight, h.channels);

		convolve3D(blockFrames, blockWidth, blockHeight, h.channels, block(0, 0), 
			   patchFrames, patchWidth, patchHeight, dimensions*h.channels, kernel,
			   out(0, 0), MATRIX);

		Window cropped(out, patchFrames/2, patchWidth/2, patchHeight/2, dt, dx, dy);

		printf("Saving block to %s of size %i x %i x %i at position %i x %i x %i\n", 
		       outputFilename, cropped.frames, cropped.width, cropped.height,
		       t*dt, x*dx, y*dy);
		SaveBlock::apply(cropped, outputFilename, t*dt, x*dx, y*dy, 0);
	    }
	}
    }

    return 0;
}
