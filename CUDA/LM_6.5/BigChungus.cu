#include <iostream>
#include <math.h>
#include <curand_kernel.h>
#include <time.h>
using namespace std;

#define N 200 // The xy Dimensions/Size of the Two Input Matrices
#define blockSize 32

/////////////// MatrixMultiply Kernel NAIVE ///////////////
// Assumes N Blocks with 32 Threads Each
__global__ void MatrixMultiplyNaive(int *A, int *B, int *C, N){

    // Need to account for different smx tid's
    int Atid = blockIdx.x * blockDim.x + threadIdx.x;
    int Btid = blockIdx.y * blockDim.y + threadIdx.y;

    // Assuming Square Matrices
    float floatingPointSum = 0.00f;

    // Do the Multiplication
    for (int i = 0; i < N; i++) {

        // Sum Each Corresponding Multiplication
        floatingPointSum = floatingPointSum + (A[(Atid * N) + i] * B[(i * N) + Btid]);

    }

    // Put the Result in the Output Array
    C[(Atid * N) + Btid] = floatingPointSum;

}

/////////////// MatrixMultiply Kernel SHARED ///////////////
// Assumes N Blocks with 32 Threads Each
__global__ void MatrixMultiplyShared(int *X, int *Y, int *Z, N){

    // Store Elements in Shared Memory Vars
    __shared__ matrixA[N];
    __shared__ matrixB[N];

    // Need to account for different smx tid's
    int Xtid = blockIdx.x * blockDim.x + threadIdx.x;
    int Ytid = blockIdx.y * blockDim.y + threadIdx.y;

    // Copy Matrix from Global to Shared Memory
    for (int i = 0; i < N; i++) {
        matrixA[i] = X[i];
        matrixB[i] = Y[i];
    }

    // Ensure Copy is Complete by Syncing
    __syncthreads();

    // Assuming Square Matrices
    float floatingPointSum = 0.00f;

    // Do the Multiplication
    for (int i = 0; i < N; i++) {

        // Sum Each Corresponding Multiplication, Using Shared Copies of Matrices 
        floatingPointSum = floatingPointSum + (matrixA[(Xtid * N) + i] * matrixB[(i * N) + Ytid]);

    }

    // Sync Again
    __syncthreads();

    // Put the Result in the Output Array
    Z[(Xtid * N) + Ytid] = floatingPointSum;

}

/////////////// MatrixAdd Kernel ///////////////
// Assumes N Blocks with 32 Threads Each
__global__ void MatrixAdd(int *C, int *Z, int *Output){

    // Need to Account for Different SMX Tid's
    int Ctid = blockIdx.x * blockDim.x + threadIdx.x;
    int Ztid = blockIdx.y * blockDim.y + threadIdx.y;

    // Assuming Square Matrices
    float floatingPointSum = 0.00f;

    // Do the Addition
    int maximumXvalue = N
    int location = maximumXvalue * Ztid + Ctid;

    // Put the Result in the Output Array
    if (location < N) {
        Output[location] = C[location] + Z[location];
    }

}

/////////////// Main ///////////////
int main(){

    int A[N * N], B[N * N], C[N * N], X[N * N], Y[N * N], Z[N * N], Output[N * N]; // Input Data: X, Y; Output Data: Z
    int *dev_A, *dev_B, *dev_C, *dev_X, *dev_Y, *dev_Z, *dev_Output; // Device Data Pointers

    // Allocate Memory on the Device/GPU
    cudaMalloc((void**)&dev_A, N*sizeof(int));
    cudaMalloc((void**)&dev_B, N*sizeof(int));
    cudaMalloc((void**)&dev_C, N*sizeof(int));
    cudaMalloc((void**)&dev_X, N*sizeof(int));
    cudaMalloc((void**)&dev_Y, N*sizeof(int));
    cudaMalloc((void**)&dev_Z, N*sizeof(int));
    cudaMalloc((void**)&dev_Output, N*sizeof(int));

    // Fill Input Arrays that are Size N x N 
    int arrayLength = N * N;
    for(int i = 0; i < arrayLength; i++){
        A[i] = curand_uniform(&localState);
        B[i] = curand_uniform(&localState);
        C[i] = curand_uniform(&localState);
        X[i] = curand_uniform(&localState);
        Y[i] = curand_uniform(&localState);
        Z[i] = curand_uniform(&localState);
        Output[i] = curand_uniform(&localState);
    }

    /////////////// Stream 1 ///////////////

    // Copy Data to the Device
    cudaMemcpyAsync(dev_A,A,N * N*sizeof(int),cudaMemcpyHostToDevice, 1);
    cudaMemcpyAsync(dev_B,B,N * N*sizeof(int),cudaMemcpyHostToDevice, 1);

    // Create Event for Timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Cuda Kernel Call
    int gridSize = (N+(blockSize-1)) / blockSize;

    // Call Event
    cudaEventRecord(start, stream = 1);
    MatrixMultiplyNaive<<<gridSize,blockSize, 0, 1>>>(dev_A, dev_B, dev_C);
    cudaEventRecord(stop);

    // Copy Memory off of the Device
    cudaMemcpyAsync(C, dev_C, N * N*sizeof(int), cudaMemcpyDeviceToHost, 3);

    // Stop Event
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    cout << "Time Elapsed: " << milliseconds << endl;

    /////////////// Stream 2 ///////////////

    // Copy Data to the Device
    cudaMemcpyAsync(dev_X,X,N * N*sizeof(int),cudaMemcpyHostToDevice, 2);
    cudaMemcpyAsync(dev_Y,Y,N * N*sizeof(int),cudaMemcpyHostToDevice, 2);

    // Create Event
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Cuda Kernel Call
    int gridSize = (N+(blockSize-1)) / blockSize;

    // Call Event
    cudaEventRecord(start, stream = 2);
    MatrixMultiplyShared<<<gridSize,blockSize, 0, 2>>>(dev_X, dev_Y, dev_Z);
    cudaEventRecord(stop);

    // Copy Memory off of the Device
    cudaMemcpyAsync(Z, dev_Z, N * N*sizeof(int), cudaMemcpyDeviceToHost, 3);

    // Stop Event
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    cout << "Time Elapsed: " << milliseconds << endl;

    /////////////// Stream 3 ///////////////

    // Copy Data to the Device
    cudaMemcpy(dev_C,C,N * N*sizeof(int),cudaMemcpyHostToDevice);
    cudaMemcpy(dev_Z,Z,N * N*sizeof(int),cudaMemcpyHostToDevice);

    // Create Event
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Cuda Kernel Call
    int gridSize = (N+(blockSize-1)) / blockSize;

    // Call Event
    cudaEventRecord(start, stream = 3);
    MatrixMultiplyShared<<<gridSize,blockSize, 0, 3>>>(dev_C, dev_Z, dev_Output);
    cudaEventRecord(stop);

    // Copy Memory off of the Device
    cudaMemcpy(Output, dev_Output, N * N*sizeof(int), cudaMemcpyDeviceToHost);

    // Stop Event
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    cout << "Time Elapsed: " << milliseconds << endl;

    // Free Memory
    cudaFree(dev_A);
    cudaFree(dev_B);
    cudaFree(dev_C);
    cudaFree(dev_X);
    cudaFree(dev_Y);
    cudaFree(dev_Z);
    cudaFree(dev_Output);
}