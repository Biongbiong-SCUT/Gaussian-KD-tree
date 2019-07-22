Gaussian KD-Trees for High-Dimensional Filtering
Andrew Adams, Natasha Gelfand, Jennifer Dolson, and Marc Levoy
SIGGRAPH '09


The included program computes a filtering on a set of input values based on their proximity in some high dimensional space. Their positions in that high dimensional space is specified by a set of position vectors.

For example, you could use this to compute a regular old Gaussian blur of an image by setting the value vectors to (r, g, b) colors from the image, and the position vectors to their (x, y) locations.

You can make this perform a bilateral filter by instead setting the positions to (x, y, r, g, b). This will mix pixels together with other pixels that are nearby in both space and color.

Prerequisites:

You'll need a linux system with cuda installed, and a fairly modern graphics card. I'm not sure if the geforce 9000 series works, but anything above it certainly does (eg GTX 260). You can get cuda from NVIDIA. It would probably be a good idea to make sure some basic cuda demos that come with cuda work before you try my code.

File formats:

These programs read and write something I've called .tmp files, because I often use them for temporary working values. They're a little space-inefficient, but very easy to read from and write to from C, matlab, etc. It's a 4D array of floats with a 16 byte header that looks like this:

struct header {
  int frames, width, height, channels;
}

For example, an 800x600 color image would have frames = 1, width = 800, height = 600, and channels = 3. A 512x512x512 grayscale volume would have frames = 512, width = 512, height = 512, and channels = 1.

Following the header is the data, in float format. The innermost dimension is channels, followed by width, then height, then frames, so a value at frame = t, width = x, height = y, and channel = c is at position (((t*height + y)*width + x)*channels + c) in the data, which is byte offset (((t*height + y)*width + x)*channels + c)*sizeof(float) + sizeof(header) in the file. Basically, the file is in scanline order, with entire frames coming one after the other.


Bilateral filtering:

Say you have an input image input.tmp, which you wish to bilateral filter with a spatial standard deviation of 5 and a color standard deviation of 0.3 (If this seems small, it's because I usually represent colors as floats between 0 and 1, you can equally use 0 to 255 and a larger color standard deviation). You would run the following:

./filter input.tmp input.tmp output.tmp 0.3 5 0

The value vectors are given by the first argument. The position vectors are given by the second. Note that these position vectors are only (r, g, b) - the program will automatically adjoin x, y, and t. T does nothing for this example.

The next three arguments are the standard deviations. First comes the color space standard deviation, then (x, y), then t. It doesn't matter what t is set to for this example, because the input has only one frame.


Non-local means:

It's a little harder to assemble the position vectors for non-local means. The positions here are patches around each input pixel. The technique, however, doesn't work so well on a huge number of dimensions - it's better to keep it under 16, so we reduce the patch vectors using PCA first like so:

./patchPCA input.tmp pca.tmp 16 5 5 5

The first argument is the input, the second is the output. Then we get the number of dimensions we wish to reduce the patches down to. Finally we have how many frames deep a patch is, and then its width and height. For denoising an image you probably want something like 1 5 5. For denoising a volume you might choose 5 5 5. If the volume is really really noisy you might want something larger than 5. There's no reason you can't make the patches really big if you want, but be aware that PCA tends to favor the low frequency information, so really big patches and not many output dimensions is similar to low-pass filtering the input.

patchPCA will save the eigenvectors it computed as separate tmp files in case you want to check them out. They usually look like kernels you might use to compute a Taylor expansion (the average value and some derivatives in various directions).

Now that we have the pca-reduced position vectors we can filter:

./filter input.tmp pca.tmp output.tmp 0.2 30 30

Other notes:

- If you get an error about not being able to find libcudart.so.2, try prepending LD_LIBRARY_PATH=. to your command line. Eg:

LD_LIBRARY_PATH=. ./filter blah blah blah

- If you get NaNs in your output, you're probably not using a large enough standard deviation. You could also try increasing the accuracy parameters, which are another three optional parameters to the filter program. Run ./filter to see what the defaults are. Try doubling them and see if that helps.

- If you compile it yourself, don't worry about all the "Advisory:..." spam from the cuda compiler. It's correctly assuming my pointers point to global memory.

- Apologies if I'm doing something stupid with cuda. patchPCA was the first cuda program I ever wrote, and filter was the second.