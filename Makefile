bilateral: Image.cpp Image.h bilateral.cpp gaussian.h gkdtree.h jpg.h png.h macros.h
	g++ -O3 bilateral.cpp Image.cpp -o bilateral  -lpng -ljpeg

