/******************************************************************************
 *cr
 *cr            (C) Copyright 2010 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ******************************************************************************/

#define BLOCK_SIZE 512
#define SIMPLE

__global__ void reduction(float *out, float *in, unsigned size)
{
    /********************************************************************
    Load a segment of the input vector into shared memory
    Traverse the reduction tree
    Write the computed sum to the output vector at the correct index
    ********************************************************************/

    // INSERT KERNEL CODE HERE
    __shared__ float sdata[BLOCK_SIZE];

    unsigned int tid = threadIdx.x;
    unsigned int start = blockIdx.x * (BLOCK_SIZE << 1);
    unsigned int i = start + tid;

    // Each thread loads up to two elements
    float sum = 0.0f;
    if (i < size)
        sum = in[i];
    if (i + BLOCK_SIZE < size)
        sum += in[i + BLOCK_SIZE];

    sdata[tid] = sum;
    __syncthreads();

    // Reduction in shared memory
    for (unsigned int s = BLOCK_SIZE >> 1; s > 0; s >>= 1) {
        if (tid < s)
            sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    // Write block's result to output
    if (tid == 0)
        out[blockIdx.x] = sdata[0];
}
