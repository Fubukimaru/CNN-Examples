Motivation:
Images as input have a large number of variables.

Convolutions reduce variables detecting patterns.
- Convolutional filters/kernels.
- Convolve the the input with a filter ( operator is \*, not a multiplication!)
- Window operand, elementwise multiplication. Do example. Andrew Ng Vertical 
    Edge detection.
- Learn the filter matrix (parameter)   
    + By this, we can learn to detect features (and higher order features)

When having a matrix of size n\*n and convolving it with a f\*f filter, we 
obtain a matrix of n-f+1 x n-f+1

Padding
-------

Two problems:

- Shrinking output
- Less importance on pixels on the edges (because of how the filters are used)

We can add additional border of P pixels to preserve the image size.

Two criteria in order to know what padding to use:

- Valid convolution: 0 padding.
- Same convolution: Pad so that output size is the same as the input size.
    + P=(f-1)/2 (n-2p-f+1 = n)
    + By convention f is usually odd (You have the sense of central pixel).


Strided convolution
-------------------

Instead of advancing the filter by one step, jumps n steps in both directions.
Output size is 1+(n\*2p-f)/s

If output size is not integer, we take the floor (Skips corners).


Convolutions over Volumes (N-Dimensional matrices)
--------------------------------------------------

Filter of size f\*f\*channels. You can detect R, G, B features, mix of colors or
don't care about colors depending on how you set the filters by channel.

Example on size:
6x6x3 * 3x3x3 = 4x4
n\*n\*channel  *  f\*f\*channel -> n-f+1 * n-f+1 * 1

Multiple filters
----------------

We can use several filters. Each filter produce a layer of the matrix. 

n\*n\*channel  *  f\*f\*channel -> n-f+1 * n-f+1 * numFilters
n\*n\*channel  *  f\*f\*channel -> n-f+1 * n-f+1 * 1
n\*n\*channel  *  f\*f\*channel -> n-f+1 * n-f+1 * 1
n\*n\*channel  *  f\*f\*channel -> n-f+1 * n-f+1 * 1
