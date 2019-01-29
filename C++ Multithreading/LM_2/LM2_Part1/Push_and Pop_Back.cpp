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
    vector<int> testVec;
    
    // Check Thread Safety
    auto lFunc = [&testVec] (int tid) {
        for (int i = 0; i < 1000; i++) {
            testVec.push_back(tid);
            testVec.pop_back();
        }
    };
    caller(lFunc);
    
    // Display Results
    for(int i = 0; i < 15; i++){
        cout << testVec[i] << endl;
    }
    
    return 0;
}
