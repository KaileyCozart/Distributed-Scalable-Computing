// LM2 - Threads and Resource Contention
// CS315 - Whitworth University
// Kailey Cozart

#include <iostream>
#include <vector>
#include <thread>

using namespace std;

template <typename T>
void caller(T func){
    
    // Create Threads
    vector<thread> myThreads;
    for(int i = 0; i < 15; i++) {
        myThreads.push_back(thread(func, i));
    }
    
    // Join Threads
    for(int i = 0; i < 15; i++) {
        myThreads[i].join();
    }
}


int main(){
    
    // Create Variables
    int VecSize = 15;
    vector<int> testVec;
    for(int i = 0; i < VecSize; i++) {
        testVec.push_back(1);
    }
    
    // [] 10 unit long test vector, spawn thread 1 at [0], spawn t2 at [1]; all doing same math operation
    
    // Check Thread Safety
    auto lFunc = [&testVec] (int tid) {
        testVec[tid] = tid * 33;
    };
    caller(lFunc);
    
    // Display Results
    for(int i = 0; i < VecSize; i++){
        cout << testVec[i] << endl;
    }
    
    return 0;
}
