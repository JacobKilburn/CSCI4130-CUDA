/******************************************************************************
 *cr
 *cr            (C) Copyright 2010 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ******************************************************************************/

__constant__ float M_c[FILTER_SIZE][FILTER_SIZE];

__global__ void convolution(Matrix N, Matrix P)
{
    /********************************************************************
    Determine input and output indexes of each thread
    Load a tile of the input image to shared memory
    Apply the filter on the input image tile
    Write the compute values to the output image at the correct indexes
    ********************************************************************/

    // INSERT KERNEL CODE HERE

        // Thread indexes
        int tx = threadIdx.x;
        int ty = threadIdx.y;
        int row_o = blockIdx.y * TILE_SIZE + ty;
        int col_o = blockIdx.x * TILE_SIZE + tx;
    
        // Allocate shared memory for the input tile
        __shared__ float N_s[TILE_SIZE + FILTER_SIZE - 1][TILE_SIZE + FILTER_SIZE - 1];
    
        // Amount of halo (padding around tile)
        int halo = FILTER_SIZE / 2;
    
        // Global coordinates of the element this thread should load
        int row_i = row_o - halo;
        int col_i = col_o - halo;
    
        // Load input tile (including halo) from global memory into shared memory
        if (row_i >= 0 && row_i < N.height && col_i >= 0 && col_i < N.width)
            N_s[ty][tx] = N.elements[row_i * N.width + col_i];
        else
            N_s[ty][tx] = 0.0f; // zero-padding for out-of-bounds
    
        __syncthreads();
    
        // Compute convolution only for valid output threads
        float output = 0.0f;
        if (ty < TILE_SIZE && tx < TILE_SIZE && row_o < P.height && col_o < P.width)
        {
            for (int i = 0; i < FILTER_SIZE; i++)
            {
                for (int j = 0; j < FILTER_SIZE; j++)
                {
                    output += M_c[i][j] * N_s[ty + i][tx + j];
                }
            }
    
            // Write the result to the output matrix
            P.elements[row_o * P.width + col_o] = output;
        }
    
}
