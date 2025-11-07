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
 
     // Thread indices within the block
     int tx = threadIdx.x;
     int ty = threadIdx.y;
 
     // Global indices in the output matrix
     int row_o = blockIdx.y * TILE_SIZE + ty;
     int col_o = blockIdx.x * TILE_SIZE + tx;
 
     // Shared memory for input tile including halo
     __shared__ float N_s[BLOCK_SIZE][BLOCK_SIZE];
 
     // Compute coordinates of corresponding input element in global memory
     int row_i = row_o - FILTER_SIZE/2;
     int col_i = col_o - FILTER_SIZE/2;
 
     // Load the shared memory tile with halo
     if ((row_i >= 0) && (row_i < N.height) && (col_i >= 0) && (col_i < N.width))
         N_s[ty][tx] = N.elements[row_i * N.width + col_i];
     else
         N_s[ty][tx] = 0.0f; // halo cells as zero
 
     __syncthreads();
 
     // Only threads that correspond to the output TILE_SIZE write output
     if (ty < TILE_SIZE && tx < TILE_SIZE && row_o < P.height && col_o < P.width)
     {
         float output = 0.0f;
 
         // Apply convolution filter on the shared memory tile
         for (int i = 0; i < FILTER_SIZE; i++)
         {
             for (int j = 0; j < FILTER_SIZE; j++)
             {
                 output += M_c[i][j] * N_s[ty + i][tx + j];
             }
         }
 
         // Write the result to global memory
         P.elements[row_o * P.width + col_o] = output;
     }
 }
 