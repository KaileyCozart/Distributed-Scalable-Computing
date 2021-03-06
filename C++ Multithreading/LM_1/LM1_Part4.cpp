// LM1 - Introduction to Threads Part IV
// CS315 - Whitworth University
// Outlined by: Scott Griffith
// Finished by: Kailey Cozart
// Modification date: September 20, 2018
//
// Threading example using a Monte Carlo algorith to approximate PI
// User will be prompted for number of threads to be used.
// Calculation of PI will be displayed.
#include <iostream>
#include <vector>
#include <thread>
#include <random> //Fancier method of generating random numbers

using namespace std;

//////////////////////////////////////////
// Random int Generation
// Developed and introduced by Kent Jones 2016
//////////////////////////////////////////
random_device rd;    // Used to produce a random seed
default_random_engine engine(rd()); // Use the defualt random number generator engine
std::uniform_real_distribution<> distribution(0, 1); // Generate a uniform real distribution between 0, 1
//
//Thread-safe C++11 pseudo-random number generator
//@return        returns a random value between 0 and 1
//
double cs273_rand() {
    return distribution(engine);
}
////////////////////////////////////////////

// withinCircle does a simple check on two coordinates to verify if they lie inside a unit circle
// Based on pythagorean theorem
// x^2 + y^2 will have to be less than 1 (radius of unit circle) to land inside
// Otherwise outside the circle, in the square
// Input: x and y are doubles between [-1 , 1]
// Output: True if within unit circle, false otherwise
bool withinCircle(double x, double y) {
    if (x*x + y * y < 1.0) {
        return true;
    }
    return false;
}

template <typename T>
//Function that is responsable for spawning threadNum threads to execute func
//Input: func is a function, threadNum is the number of threads that should be spawned
//Output: Threads are executed, and joined after executing
void caller(T func, int threadNum) { //If you would like to change the parameters you can. Be sure to document changes
    
    //Student Work:
    //Develop a caller function that spawns threads equal to threadNum
    //This can be a direct copy of your caller from part 3
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
    int numberOfPoints = 1e8; // For testing purposes, this can be changed.
    int numThreads; // Number of threads to be used
    float answer; // Put your answer in here!
    
    // arbitrary thread count! Get User defined threads
    cout << "How many threads are we going to use? ";
    cin >> numThreads;
    
    // Start Student Code:
    
    // You are going to need some places to store intermediate data
    vector<int> intermediateData(numThreads); // initialize size of vector...
    // Write a function, I would suggest a lambda function to run a subset of the total calculations
    // Each thread will have an iteration over a bunch of points (numberOfPoints/numThreads)
    // Each iteration will assign a random number (cs273_rand()) to an x and y value
    // Check to see values are in the circle (withinCircle(x,y)
    // Update variables
    auto lFunc = [&intermediateData, &numThreads, &numberOfPoints](int tid) {
        int numberOfPtsPerThread = (numberOfPoints/numThreads);
        int counterInCircle = 0;
        for(int i = 0; i < numberOfPtsPerThread; i++) {
            double x = cs273_rand();
            double y = cs273_rand();
            if (withinCircle(x, y)) {
                counterInCircle++;
            }
        }
        intermediateData[tid] = counterInCircle;
    };
    
    // Run your function using caller()
    caller(lFunc, numThreads);
    // Accumulate each thread's results and calculate PI
    // pi = 4 * (number of points in the circle/number of points in the square)
    float sum = 0.0;
    for(int i = 0; i < numThreads; i++) {
        sum += intermediateData[i];
    }
    answer = 4.0 * (sum / numberOfPoints);
    // End Student Code
    
    cout << "Approximate value of pi is: " << answer << endl;
}
