#include <iostream>
using namespace std;

#define N 65536
#define A 2
#define blockSize 65

// SAXPY Kernel
// Performs A*X+Y
// Assumes single N blocks with 32 threads each
__global__ void saxpy(int *X, int *Y, int *Z){
    // Need to account for different smx tid's
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid<N){
        Z[tid] = A * X[tid] + Y[tid];
    }
}

int main(){

    int X[N], Y[N], Z[N]; // Host data: X,Y input data, Z output data
    int *dev_X, *dev_Y, *dev_Z; // Device data pointers

    // Allocate memory on the device/GPU
    cudaMalloc((void**)&dev_X, N*sizeof(int));
    cudaMalloc((void**)&dev_Y, N*sizeof(int));
    cudaMalloc((void**)&dev_Z, N*sizeof(int));

    // Fill input arrays
    for(int i = 0; i<N; i++){
        X[i] = i;
        Y[i] = i*i;
    }

    // Copy data to the device
    cudaMemcpy(dev_X,X,N*sizeof(int),cudaMemcpyHostToDevice);
    cudaMemcpy(dev_Y,Y,N*sizeof(int),cudaMemcpyHostToDevice);

    // Create Event
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Cuda Kernel Call
    int gridSize = (N+(blockSize-1)) / blockSize;
    // (N+31) / 32
    // Call Event
    cudaEventRecord(start);
    saxpy<<<gridSize,blockSize>>>(dev_X,dev_Y,dev_Z);
    cudaEventRecord(stop);

    // Copy memory off of the device
    cudaMemcpy(Z,dev_Z,N*sizeof(int),cudaMemcpyDeviceToHost);

    // Stop Event
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    cout << milliseconds << endl;

    // Check out contents of working arrays/output data
    for(int i = 0; i<N; i++){
        int checkValue = A * X[i] + Y[i];
        if (Z[i] != checkValue) {
            cout << "Mismatch " << i << endl;
        }
    }

    // Free up memory
    cudaFree(dev_X);
    cudaFree(dev_Y);
    cudaFree(dev_Z);
}