// transpose_parallel_per_element_tiled 8x8: 0.229056 ms.
// transpose_parallel_per_element_tiled 16x16: 0.130112 ms.
// transpose_parallel_per_element_tiled 32x32: 0.34448 ms.
// transpose_parallel_per_element 8x8: 0.227008 ms.
// transpose_parallel_per_element 16x16: 0.172192 ms.
// transpose_parallel_per_element 32x32: 0.344384 ms.
#include <stdio.h>
#include "gputimer.h"
#include "utils.h"

const int N= 1024;	// matrix size will be NxN
const int K= 16;		// TODO, set K to the correct value and tile size will be KxK


// to be launched with one thread per element, in (tilesize)x(tilesize) threadblocks
// thread blocks read & write tiles, in coalesced fashion
// adjacent threads read adjacent input elements, write adjacent output elmts
__global__ void 
transpose_parallel_per_element_tiled(float in[], float out[])
{
	// TODO
	int i = blockIdx.y * blockDim.y + threadIdx.y;
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	int idx = i * N + j;
	extern __shared__ float in_sh[];
	in_sh[threadIdx.x * blockDim.x + threadIdx.y] = in[idx];
	__syncthreads();
	
	// out[j * N + i] = in[i * N + j]
	int i1 = blockIdx.x * blockDim.x + threadIdx.y;
	int j1 = blockIdx.y * blockDim.y + threadIdx.x;
	out[i1 * N + j1] = in_sh[threadIdx.y * blockDim.x + threadIdx.x];
}

// The following functions and kernels are for your references
void 
transpose_CPU(float in[], float out[])
{
	for(int j=0; j < N; j++)
    	for(int i=0; i < N; i++)
      		out[j + i*N] = in[i + j*N]; // out(j,i) = in(i,j)
}

// to be launched on a single thread
__global__ void 
transpose_serial(float in[], float out[])
{
	for(int j=0; j < N; j++)
		for(int i=0; i < N; i++)
			out[j + i*N] = in[i + j*N]; // out(j,i) = in(i,j)
}

// to be launched with one thread per row of output matrix
__global__ void 
transpose_parallel_per_row(float in[], float out[])
{
	int i = threadIdx.x;

	for(int j=0; j < N; j++)
		out[j + i*N] = in[i + j*N]; // out(j,i) = in(i,j)
}

// to be launched with one thread per element, in KxK threadblocks
// thread (x,y) in grid writes element (i,j) of output matrix 
__global__ void 
transpose_parallel_per_element(float in[], float out[])
{
	int i = blockIdx.x * K + threadIdx.x;
	int j = blockIdx.y * K + threadIdx.y;

	out[j + i*N] = in[i + j*N]; // out(j,i) = in(i,j)
}

int main(int argc, char **argv)
{
	int numbytes = N * N * sizeof(float);

	float *in = (float *) malloc(numbytes);
	float *out = (float *) malloc(numbytes);
	float *gold = (float *) malloc(numbytes);

	fill_matrix(in, N);
	transpose_CPU(in, gold);

	float *d_in, *d_out;

	cudaMalloc(&d_in, numbytes);
	cudaMalloc(&d_out, numbytes);
	cudaMemcpy(d_in, in, numbytes, cudaMemcpyHostToDevice);

	GpuTimer timer;

/*  
 * Now time each kernel and verify that it produces the correct result.
 *
 * To be really careful about benchmarking purposes, we should run every kernel once
 * to "warm" the system and avoid any compilation or code-caching effects, then run 
 * every kernel 10 or 100 times and average the timings to smooth out any variance. 
 * But this makes for messy code and our goal is teaching, not detailed benchmarking.
 */

	dim3 blocks(N / K,N / K);	//TODO, you need to set the proper blocks per grid
	dim3 threads(K,K);	//TODO, you need to set the proper threads per block

	timer.Start();
	//transpose_parallel_per_element_tiled<<<blocks, threads, K * K * sizeof(float)>>>(d_in, d_out);
	transpose_parallel_per_element<<<blocks, threads>>>(d_in, d_out);
	timer.Stop();
	cudaMemcpy(out, d_out, numbytes, cudaMemcpyDeviceToHost);
	printf("transpose_parallel_per_element_tiled %dx%d: %g ms.\nVerifying ...%s\n", 
		   K, K, timer.Elapsed(), compare_matrices(out, gold, N) ? "Failed" : "Success");

	cudaFree(d_in);
	cudaFree(d_out);
}
