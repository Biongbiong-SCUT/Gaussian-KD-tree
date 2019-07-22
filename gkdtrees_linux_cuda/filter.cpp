#include "gpu_gkdtree.h"
#include <stdlib.h>
#include <stdio.h>
#include <time.h>

#include <ImageStack.h>

using namespace ImageStack;

struct header {
    int frames, width, height, channels;
};

header peekAtHeader(const char *filename) {
    FILE *f = fopen(filename, "rb");
    header h;
    fread(&h, sizeof(header), 1, f);
    fclose(f);
    return h;
}

void getReference(char *filename, 
		  float dataScale, float spaceScale, float timeScale, 
		  int t, int x, int y, Image out) {
    for (int dt = 0; dt < out.frames; dt++) {
	Image refBlock = LoadBlock::apply(filename, t+dt, x, y, 0, 1, out.width, out.height, -1);
	for (int dy = 0; dy < refBlock.height; dy++) {
	    for (int dx = 0; dx < refBlock.width; dx++) {
		float *outPtr = out(dt, dx, dy);
		float *blkPtr = refBlock(dx, dy);
		outPtr[0] = (dx+x) * spaceScale;
		outPtr[1] = (dy+y) * spaceScale;
		outPtr[2] = (dt+t) * timeScale;
		for (int c = 0; c < refBlock.channels; c++) {
		    outPtr[c+3] = blkPtr[c] * dataScale;
		}
	    }
	}
    }
}

// if you want to increase the accuracy of the result, scale up these numbers
#define SPLAT_ACCURACY 128
#define BLUR_ACCURACY 256
#define SLICE_ACCURACY 64


#define SPLAT_STDDEV 0.3015113446
#define BLUR_STDDEV 0.9045340337
#define SLICE_STDDEV 0.3015113446


int main(int argc, char **argv) {
    if (argc < 7) {
      printf("Usage: ./filter input.tmp reference.tmp output.tmp <data stdev> <xy stdev> <z stddev> <splatAccuracy (defaults to %d)> <blur accuracy (defaults to %d)> <slice accuracy (defaults to %d)>\n", SPLAT_ACCURACY, BLUR_ACCURACY, SLICE_ACCURACY);
	return 1;
    }

    srand(time(NULL));

    initCuda(1, argv);

    header imHeader = peekAtHeader(argv[1]);
    header refHeader = peekAtHeader(argv[2]);

    if (imHeader.frames != refHeader.frames ||
	imHeader.width != refHeader.width || 
	imHeader.height != refHeader.height) {
	printf("image and reference are of different sizes\n");
	return 1;
    }

    int size = imHeader.width*imHeader.height*imHeader.frames;

    float dataStdev = atof(argv[4]);
    float spaceStdev = atof(argv[5]);
    float timeStdev = atof(argv[6]);

    float dataScale = dataStdev > 0 ? (1.0/dataStdev) : 1000;
    float spaceScale = spaceStdev > 0 ? (1.0/spaceStdev) : 1000;
    float timeScale = timeStdev > 0 ? (1.0/timeStdev) : 1000;

    int splatAccuracy = SPLAT_ACCURACY;
    int blurAccuracy = BLUR_ACCURACY;
    int sliceAccuracy = SLICE_ACCURACY;

    if (argc == 10) {
	splatAccuracy = atoi(argv[7]);
	blurAccuracy = atoi(argv[8]);
	sliceAccuracy = atoi(argv[9]);
    }

    int spaceMargin = (int)(spaceStdev*2+1);
    int timeMargin = (int)(timeStdev*2+1);

    // figure out how to block the volume
    int xBlocks = 1, yBlocks = 1, tBlocks = 1;
    long long blockWidth, blockHeight, blockFrames;
    int blockXDelta, blockYDelta, blockTDelta;
    while (1) {
	// figure out the block sizes 
	blockXDelta = ((imHeader.width-1)/xBlocks + 1);
	blockWidth = blockXDelta + 2*spaceMargin;
	blockYDelta = ((imHeader.height-1)/yBlocks + 1);
	blockHeight = blockYDelta + 2*spaceMargin;
	blockTDelta = ((imHeader.frames-1)/tBlocks + 1);
	blockFrames = blockTDelta + 2*timeMargin;
	if (blockFrames > imHeader.frames) {
	    blockFrames = imHeader.frames;
	}
	if (blockWidth > imHeader.width) {
	    blockWidth = imHeader.width;
	}
	if (blockHeight > imHeader.height) {
	    blockHeight = imHeader.height;
	}

	long long memory = blockWidth*blockHeight*blockFrames;
	// imHeader.channels+1 for each value
	// refHeader.channels+3 for each position
	// 8 for each inner node
	// 4 for each entry in each build queue
	memory *= (imHeader.channels+1 + refHeader.channels+3 + 8 + 4*2)*4;
	printf("Scheme %i %i %i uses %lli memory\n", tBlocks, xBlocks, yBlocks, memory);
	if (memory < (1ull << 31)) break;

	// try increasing each one by 1, and see which works the best
	long long newWorkx  = spaceMargin*blockFrames*blockHeight;
	long long newWorky  = spaceMargin*blockFrames*blockWidth;
	long long newWorkt  = timeMargin*blockWidth*blockHeight;
	if (newWorkx < newWorky && newWorkx < newWorkt) {
	    xBlocks++;
	} else if (newWorky < newWorkt) {
	    yBlocks++;
	} else {
	    tBlocks++;
	}
    }

    if (blockHeight == imHeader.height && blockWidth == imHeader.width) spaceMargin = 0;
    if (blockFrames == imHeader.frames) timeMargin = 0;

    printf("Blocking chosen: %i %i %i\n", tBlocks, xBlocks, yBlocks);

    CreateTmp::apply(argv[3], imHeader.frames, imHeader.width, imHeader.height, imHeader.channels);

    printf("Output created... building tree\n"); fflush(stdout);

    for (int bt = 0; bt < tBlocks; bt++) {
	int minT = bt * blockTDelta - timeMargin;
	int maxT = minT + blockTDelta + 2*timeMargin;
	int outMinT = bt * blockTDelta;
	int outSizeT = blockTDelta;
	if (minT < 0) minT = 0;
	if (maxT > imHeader.frames) maxT = imHeader.frames;
	int sizeT = maxT - minT;
	if (outMinT + outSizeT > imHeader.frames) 
	    outSizeT = imHeader.frames - outMinT;
	for (int by = 0; by < yBlocks; by++) {
	    int minY = by * blockYDelta - spaceMargin;
	    int maxY = minY + blockYDelta + 2*spaceMargin;
	    int outMinY = by * blockYDelta;
	    int outSizeY = blockYDelta;
	    if (minY < 0) minY = 0;
	    if (maxY > imHeader.height) maxY = imHeader.height;
	    int sizeY = maxY - minY;
	    if (outMinY + outSizeY > imHeader.height)
		outSizeY = imHeader.height - outMinY;
	    for (int bx = 0; bx < xBlocks; bx++) {
		int minX = bx * blockXDelta - spaceMargin;
		int maxX = minX + blockXDelta + 2*spaceMargin;
		int outMinX = bx * blockXDelta;
		int outSizeX = blockXDelta;
		if (minX < 0) minX = 0;
		if (maxX > imHeader.width) maxX = imHeader.width;
		int sizeX = maxX - minX;
		if (outMinX + outSizeX > imHeader.width)
		    outSizeX = imHeader.width - outMinX;

		printf("Block input is (%i %i %i) to (%i %i %i)\n",
		       minT, minX, minY,
		       minT + sizeT, minX + sizeX, minY + sizeY);

		printf("Block output is (%i %i %i) to (%i %i %i)\n",
		       outMinT, outMinX, outMinY,
		       outMinT + outSizeT, outMinX + outSizeX, outMinY + outSizeY);

		printf("Loading reference block... \n"); fflush(stdout);

		int size = sizeT*sizeX*sizeY;
		       
		Image refFrame(sizeT, sizeX, sizeY, refHeader.channels+3);
		getReference(argv[2], dataScale, spaceScale, timeScale, minT, minX, minY, refFrame);

		printf("Loading image block... \n"); fflush(stdout);

		Image outFrame(sizeT, sizeX, sizeY, imHeader.channels+1);
		Image imFrame = LoadBlock::apply(argv[1], minT, minX, minY, 0, sizeT, sizeX, sizeY, -1);
		Window outWindow(imFrame, outMinT-minT, outMinX-minX, outMinY-minY,
				 outSizeT, outSizeX, outSizeY);
		
		buildTree(refFrame.channels, imFrame.channels, refFrame(0, 0), size, size);

		printf("Done constructing tree... computing internal bounds\n"); fflush(stdout);
		finalizeTree();   
		printf("Splatting...\n"); fflush(stdout);
		
		gaussianSplat(refFrame(0, 0), imFrame(0, 0),
			      size, splatAccuracy, SPLAT_STDDEV); 

		printf("Done splatting... blurring\n"); fflush(stdout);
		gaussianBlur(blurAccuracy, BLUR_STDDEV);
		printf("Done blurring... slicing\n"); fflush(stdout);
		
		gaussianSlice(refFrame(0, 0), outFrame(0, 0),
			      size, sliceAccuracy, SLICE_STDDEV);
		
		for (int t = outMinT - minT; t < outMinT - minT + outSizeT; t++) {
		    for (int y = outMinY - minY; y < outMinY - minY + outSizeY; y++) {
			for (int x = outMinX - minX; x < outMinX - minX + outSizeX; x++) {
			    double mult = 1.0/outFrame(t, x, y)[imHeader.channels];
			    for (int c = 0; c < imHeader.channels; c++) {
				imFrame(t, x, y)[c] = outFrame(t, x, y)[c]*mult;
			    }
			}
		    }
		}

		printf("Writing output...\n"); fflush(stdout);
		SaveBlock::apply(outWindow, argv[3], outMinT, outMinX, outMinY, 0);

		destroyTree();
	    }
	}
    }

    fflush(stdout);
    return 0;
}
