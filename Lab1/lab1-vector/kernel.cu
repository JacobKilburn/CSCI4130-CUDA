/******************************************************************************
 *cr
 *cr            (C) Copyright 2010 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ******************************************************************************/

 __global__ void vecAddKernel(float *A, float *B, float *C, int n)
 {
	 // Calculate global thread index based on the block and thread indices ----
	 //INSERT KERNEL CODE HERE
 
	 // Use global index to determine which elements to read, add, and write ---
	 //INSERT KERNEL CODE HERE, BE CAREFUL FOR CORNER CASE!!!
	 
	 // int i = threadIdx.x;
	 // C[i] = A[i] + B[i];
 
	 //what is n used for? For loop?
 
	 int i = blockIdx.x * blockDim.x + threadIdx.x;
	 if (i<n) C[i] = A[i] + B[i];
 }
 
 __global__ void image2grayKernel(float *in, float *out, int height, int width)
 {
	 // Calculate global thread index based on the block and thread indices ----
	 //INSERT KERNEL CODE HERE
	 
	 // Use global index to determine which elements to read, add, and write ---
	 //INSERT KERNEL CODE HERE, BE CAREFUL FOR CORNER CASE!!!
	 
	 int x = blockIdx.x * blockDim.x + threadIdx.x; // column
	 int y = blockIdx.y * blockDim.y + threadIdx.y; // row
 
	 if (x < width && y < height) {
		 int i = y * width + x;
		 int rgb_idx = i * 3;
 
		 float r = in[rgb_idx];
		 float g = in[rgb_idx + 1];
		 float b = in[rgb_idx + 2];
 
		 out[i] = 0.144f * r + 0.587f * g + 0.299f * b;
	 }
 }