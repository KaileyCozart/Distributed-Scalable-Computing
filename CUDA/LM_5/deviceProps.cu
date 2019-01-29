#include <iostream>
using namespace std;

int main() {
    	
    // Get the Number of Devices
    int count;
    cudaGetDeviceCount(&count);
    cout << "Number of Devices: " << count << endl;

    // Get Useful Properties
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    // Output Requested Information
    cout << "Name of Device: " << prop.name << endl;
    cout << "Global Memory Capacity: " << prop.totalGlobalMem << endl;
    cout << "Shared Memory Maximum: " << prop.sharedMemPerBlock << endl;
    cout << "Warp Size: " << prop.warpSize << endl;
    cout << "Maximum Threads per Block: " << prop.maxThreadsPerBlock << endl;
    cout << "Maximum Dimensions for Thread Blocks: (" << prop.maxThreadsDim[0];
    cout << ", " << prop.maxThreadsDim[1] << ", " << prop.maxThreadsDim[2] <<")" << endl;
    cout << "Maximum Dimensions for Grid of Blocks: (" << prop.maxGridSize[0];
    cout << ", " << prop.maxGridSize[1] << ", " << prop.maxGridSize[2] << ")" << endl;
    cout << "Compute Capability: " << prop.major << "." << prop.minor << endl;

    // Output Additional Information
    cout << "Clock Speed in KHz: " << prop.memoryClockRate << endl;
    cout << "Width of Memory Bus: " << prop.memoryBusWidth << endl;    

}