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
    vector<int> testVector;
    testVector.push_back(33);
    
    // Check Thread Safety
    auto lFunc = [&testVector] (int tid) {
        testVector.insert(testVector.begin(), 5);
    };
    caller(lFunc);
    
    // Display Results
    for(int i = 0; i < testVector.size(); i++){
        cout << testVector[i] << endl;
    }
    
    return 0;
}
