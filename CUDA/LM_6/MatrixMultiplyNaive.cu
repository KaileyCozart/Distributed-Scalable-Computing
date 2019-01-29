#include <iostream>
#include <math.h>
#include <curand_kernel.h>
#include <time.h>
using namespace std;

#define N 200 // The xy Dimensions/Size of the Two Input Matrices
#define blockSize 32

// MatrixMultiply Kernel
// Assumes N Blocks with 32 Threads Each
__global__ void MatrixMultiply(int *X, int *Y, int *Z, N){

    // Need to account for different smx tid's
    int Xtid = blockIdx.x * blockDim.x + threadIdx.x;
    int Ytid = blockIdx.y * blockDim.y + threadIdx.y;

    // Assuming Square Matrices
    float floatingPointSum = 0.00f;

    // Do the Multiplication
    for (int i = 0; i < N; i++) {

        // Sum Each Corresponding Multiplication
        floatingPointSum = floatingPointSum + (X[(Xtid * N) + i] * Y[(i * N) + Ytid]);

    }

    // Put the Result in the Output Array
    Z[(Xtid * N) + Ytid] = floatingPointSum;

}

int main(){

    int X[N], Y[N], Z[N]; // Input Data: X, Y; Output Data: Z
    int *dev_X, *dev_Y, *dev_Z; // Device Data Pointers

    // Allocate Memory on the Device/GPU
    cudaMalloc((void**)&dev_X, N*sizeof(int));
    cudaMalloc((void**)&dev_Y, N*sizeof(int));
    cudaMalloc((void**)&dev_Z, N*sizeof(int));

    // Fill Input Arrays that are Size N x N 
    int arrayLength = N * N;
    for(int i = 0; i < arrayLength; i++){
        X[i] = curand_uniform(&localState);
        Y[i] = curand_uniform(&localState);
        Z[i] = curand_uniform(&localState);
    }

    // Copy Data to the Device
    cudaMemcpy(dev_X,X,N*sizeof(int),cudaMemcpyHostToDevice);
    cudaMemcpy(dev_Y,Y,N*sizeof(int),cudaMemcpyHostToDevice);

    // Create Event
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Cuda Kernel Call
    int gridSize = (N+(blockSize-1)) / blockSize;

    // Call Event
    cudaEventRecord(start);
    MatrixMultiply<<<gridSize,blockSize>>>(dev_X, dev_Y, dev_Z, sN);
    cudaEventRecord(stop);

    // Copy Memory off of the Device
    cudaMemcpy(Z, dev_Z, N*sizeof(int), cudaMemcpyDeviceToHost);

    // Stop Event
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    cout << "Time Elapsed: " << milliseconds << endl;

    // Check Contents of Working Arrays/Output Data
    int checkValue;
    for (int j = 0; j < arrayLength; j++) {
        for (int i = 0; i < N; i++) { // Loop for Checking Each Value
            checkValue = checkValue + (X[(i * N) + i] * Y[(i * N) + i]);
        }
        if (Z[i] != checkValue) {
            cout << "Mismatch " << i << endl;
        }
    }

    // Free Memory
    cudaFree(dev_X);
    cudaFree(dev_Y);
    cudaFree(dev_Z);
}