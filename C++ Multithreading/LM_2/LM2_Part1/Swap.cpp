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
    int maxVal = 15;
    vector<int> testVec;
    for(int i = 0; i < maxVal; i++) {
        testVec.push_back(i);
    }
    
    vector<int> swapVec;
    for(int i = 0; i < maxVal; i++) {
        swapVec.push_back(66);
    }
    
    // Check Thread Safety
    auto lFunc = [&testVec, &swapVec] (int tid) {
        testVec.swap(swapVec);
    };
    caller(lFunc);
    
    // Display Results
    for(int i = 0; i < maxVal; i++){
        cout << testVec[i] << endl;
    }
    
    return 0;
}
