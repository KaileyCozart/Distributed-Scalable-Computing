#include <iostream>
#include <math.h>
#include <curand_kernel.h>
#include <time.h>
using namespace std;

#define N 1024
#define R 4

/////////////// Parallel Averaging Kernel ///////////////
__global__ void Average(int *input) {
    
    int tid = threadIdx.x;
    int threadsPerBlock = blockDim.x;
    int gapSize = 1; // How Big the Gap Btwn Valid Elements Will Be

    while (threadsPerBlock > 0) { // Work Bckwrds

        if (tid < threadsPerBlock) { 
            int A = tid * gapSize * R;
            int B = first + gapSize;
            input[A] = input[A] + input[B];
        }

        // Change Gap and Threads per Block for Next Iteration
        gapSize = gapSize * R;
		threadsPerBlock = threadsPerBlock / R;
    }

    input[0] = input[0] / (threadsPerBlock * R);
}

/////////////// Main Function ///////////////
int main(){

    srand(N);

    // Allocate Space for Array on Device
	int array[N];
    int *dev_A; 
    cudaMalloc((void**)&dev_A, N * sizeof(int));

    // Put Values into Input Array
	for(int i = 0; i < N; i++) {
	    array[i] = rand();
	}

    int dataSize = N * sizeof(int); // Size of Data
    
	cudaMemcpy(dev_A, array, dataSize, cudaMemcpyHostToDevice);
    Average<<<1,N/R>>>(array); // Kernel Call
	cudaMemcpy(&array, dev_A, sizeof(int), cudaMemcpyDeviceToHost);
   
    // Free Memory
    cudaFree(array);
    
    return 0;
}