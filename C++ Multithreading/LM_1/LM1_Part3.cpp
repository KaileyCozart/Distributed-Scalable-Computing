//LM1 - Introduction to Threads Part III
// CS315 - Whitworth University
// Outlined by: Scott Griffith
// Finished by: Kailey Cozart
// Modification date: September 20, 2018
//
// Threading example averging a large vector of random ints.
// User will be prompted for number of threads to be used.
// Average will be displayed.
#include <iostream>
#include <vector>
#include <thread>
#include <time.h> //Used by fillRandom to get a variable seed
#include <numeric> //Need for the accumulate function

using namespace std;

//Function to randomly fill an integer vector with values
//All values will be between 0 and 99
//Should average to 49.5
//Input: vec will be filled with size number of random ints. Assumes vec empty.
//Output: vec passed back by reference, filled with size integers
void fillRandom(vector<int>& vec, int size) { //You should not change this function. Leave as is.
    srand(time(NULL)); //Set random seed based on current time
    for (int i = 0; i < size; i++) { //Iterate from 0 to size
        vec.push_back((rand() % 100)); //Add random number between 0 and 99
    }
}

template <typename T>
//Function that is responsable for spawning threadNum threads to execute func
//Input: func is a function, threadNum is the number of threads that should be spawned
//Output: Threads are executed, and joined after executing
void caller(T func, int threadNum) { //If you would like to change the parameters you can. Be sure to document changes
    
    //Student Work:
    //Develop a caller function that spawns threads equal to threadNum
    //You may want to build off the work done in part 2
    //Hint: vector<thread> is your friend
    vector<thread> myThreads;
    for (int i = 0; i < threadNum; i++) {
        myThreads.push_back(thread(func, i));
    }
    for (int i = 0; i < threadNum; i++) {
        myThreads[i].join();
    }
}

int main() {
    vector<int> problem; //Vector that needs to be averaged
    fillRandom(problem, 100000); //Fill problem with 100,000 random integers
    //Feel free to change the number passed to fillRandom. This might be necessary if you are looking for differences in processing time
    float answer = 0.0; //Put your answer in here. Pay attention to types here. Just because you are averaging ints, your answer (and all intermediate calculations) should be float
    int numThreads; //Number of threads to be used
    
    //arbitrary thread count! Get User defined threads
    cout << "How many threads are we going to use? ";
    cin >> numThreads;
    
    //Student Work Below:
    //Call caller function that is passed numThreads and the function you want processed (can be Lambda)
    // Stop divisible
    
    float threadAnswers[numThreads];
    
    auto lFunc = [&problem, &numThreads, &threadAnswers](int tid) {
        int threadRange = 100000 / numThreads;
        int beginningRange = tid * threadRange;
        int endRange;
        if (tid == (numThreads - 1)) {
            endRange = problem.size();
        }
        else {
            endRange = (tid + 1) * threadRange;
        }
        float threadResult = 0.0;
        for (int i = beginningRange; i < endRange; i++) {
            threadResult += problem[i];
        }
        float threadAverage = threadResult / threadRange;
        threadAnswers[tid] = threadAverage;
    };
    
    caller(lFunc, numThreads);
    
    float sum = 0.0;
    for(int i = 0; i < numThreads; i++) {
        sum += threadAnswers[i];
    }
    answer = sum / numThreads;
    
    //Student work end
    
    cout << "The average is: " << answer << " calculated using " << numThreads << " threads." << endl; //Report your findings
}
