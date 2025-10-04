/******************************************************************************
 *cr
 *cr            (C) Copyright 2010 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ******************************************************************************/

#include <stdio.h>

// Feel free to use other numbers for best performance
#define TILE_SIZE 16

__global__ void mysgemm(int m, int n, int k, const float *A, const float *B, float *C)
{

    /********************************************************************
     *
     * Compute C = A x B
     *   where A is a (m x k) matrix
     *   where B is a (k x n) matrix
     *   where C is a (m x n) matrix
     *
     * Use shared memory for tiling
     *
     ********************************************************************/

    // INSERT KERNEL CODE HERE
    // Compute thread row and column within the output matrix
    int Row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int Col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float Cvalue = 0.0f;

    // Shared memory tiles for A and B
    __shared__ float Asub[TILE_SIZE][TILE_SIZE];
    __shared__ float Bsub[TILE_SIZE][TILE_SIZE];

    // Loop over tiles of A and B needed to compute C[Row][Col]
    for (int t = 0; t < (k + TILE_SIZE - 1) / TILE_SIZE; ++t)
    {
        // Load data into shared memory if within bounds
        if (Row < m && t * TILE_SIZE + threadIdx.x < k)
            Asub[threadIdx.y][threadIdx.x] = A[Row * k + t * TILE_SIZE + threadIdx.x];
        else
            Asub[threadIdx.y][threadIdx.x] = 0.0f;

        if (Col < n && t * TILE_SIZE + threadIdx.y < k)
            Bsub[threadIdx.y][threadIdx.x] = B[(t * TILE_SIZE + threadIdx.y) * n + Col];
        else
            Bsub[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        // Multiply the tiles together
        for (int i = 0; i < TILE_SIZE; ++i)
            Cvalue += Asub[threadIdx.y][i] * Bsub[i][threadIdx.x];

        __syncthreads();
    }

    // Store the result
    if (Row < m && Col < n)
        C[Row * n + Col] = Cvalue;



}

void basicSgemm(char transa, char transb, int m, int n, int k, float alpha, const float *A, int lda, const float *B, int ldb, float beta, float *C, int ldc, int testRound)
{
    if ((transa != 'N') && (transa != 'n'))
    {
        printf("unsupported value of 'transa'\n");
        return;
    }

    if ((transb != 'N') && (transb != 'n'))
    {
        printf("unsupported value of 'transb'\n");
        return;
    }

    if ((alpha - 1.0f > 1e-10) || (alpha - 1.0f < -1e-10))
    {
        printf("unsupported value of alpha\n");
        return;
    }

    if ((beta - 0.0f > 1e-10) || (beta - 0.0f < -1e-10))
    {
        printf("unsupported value of beta\n");
        return;
    }

    // Initialize thread block and kernel grid dimensions ----------------------
    // INSERT CODE HERE
    dim3 blockDim(TILE_SIZE, TILE_SIZE);
    dim3 gridDim((n + TILE_SIZE - 1) / TILE_SIZE, (m + TILE_SIZE - 1) / TILE_SIZE);




    for (int i = 0; i < testRound; i++) {
        // Invoke CUDA kernel --------------------------------------------------
        // INSERT CODE HERE
        mysgemm<<<gridDim, blockDim>>>(m, n, k, A, B, C);



        cudaDeviceSynchronize();
    }
}
