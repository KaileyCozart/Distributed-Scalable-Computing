#include <iostream>
#include <thread>
#include <vector>
using namespace std;

void helloThreads(int tid, int x) {
    cout << "Konichiwa! I am thread " << tid << " with a number " << x << endl;
}

int main() {
    
    //Spawn New Threads
    vector<thread> hellos;
    for(int i = 0; i < 10; i++) {
        hellos.push_back(thread(helloThreads, i, i*2));
    }

    // Do Stuff in Main Threads

    // Join Spawned Threads Back into Main Thread
    for(int i = 0; i < 10; i++) {
        hellos[i].join();
    }

    return 0;
}