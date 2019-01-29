#include <thread>
#include <condition_variable>
#include <mutex>
#include <vector>
#include <iostream>

using namespace std;

condition_variable_any flip_cv;     // Needs mutext to work correctly
mutex mtx;                          // Mutex will be used by condition_variable_any as the lock
bool whatAmI;
int shuffleCount = 0;
int sortCount = 0;

// **Delay Method Built By: Scott**
void delay(){
    for(int i = 0; i < 1000000; i++){
        for(int i = 0; i < 1000; i++){
            float x = 2e9; float y = 1;
            y = y*x;
        }
    }
}

// **RNG Struct Built By: bames53 on Stack Overflow ;)**
struct RNG {
    int operator() (int n) {
        return std::rand() / (1.0 + RAND_MAX) * n;
    }
};

void myShuffle(vector<int> vector){
    while(1){
        delay(); // Let other thread get to wait.
        whatAmI = true;
        
        if (shuffleCount < 5) {
            srand(time(0));
            random_shuffle(vector.begin(), vector.end(), RNG());
            // random_shuffle (vector.begin(), vector.end()); // Shuffle the stuffs
            cout << "Finished shuffle #" << shuffleCount + 1 << endl;
            for(int i = 0; i < 10; i++)
            {
                cout << vector[i] << " ";
            }
            cout << endl;
            shuffleCount++;
        }
        else {
            return;
        }
        
        flip_cv.notify_one(); // Tell mySort I am done, flip_cv must already be waiting for notify_one to have any effect
        flip_cv.wait(mtx); // Wait for mySort to tell me I am done
    }
}

void mySort(vector<int> vector){
    while(1){
        flip_cv.wait(mtx); // Wait for myShuffle to tell me I am done
        
        delay(); // Let other thread get to wait.
        whatAmI = false;
        
        if (sortCount < 5) {
            sort (vector.begin(), vector.begin()+10); // Sort the stuffs
            cout << "Finished sort #" << sortCount + 1 << endl;
            for(int i = 0; i < 10; i++)
            {
                cout << vector[i] << " ";
            }
            cout << endl;
            sortCount++;
        }
        else {
            return;
        }
        
        flip_cv.notify_one(); // Tell myShuffle I am done, flip_cv must already be waiting for notify_one to have any effect
    }
}

int main(){
    int vectorSize = 10;
    vector<int> vector;
    for (int i = 0; i < vectorSize; i++) {
        vector.push_back(i);
    }
    
    thread one = thread(myShuffle, vector);
    thread two = thread(mySort, vector);
    
    one.join();
    two.join();
    
    return 0;
}
