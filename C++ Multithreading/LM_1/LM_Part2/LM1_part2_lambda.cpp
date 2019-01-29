// LM1 - Introduction to Threads Part II
// CS315 - Whitworth University
// Outlined by: Scott Griffith
// Finished by: Kailey Cozart
// Modification date: September 20, 2018
//
// Basic Lambda function Threading example
#include <iostream>
#include <vector>
#include <thread>

using namespace std;

template <typename T>
//Function to spawn ten threads processing func
//Each thread is initilized by passing a unique integer value (0-9) into the first argument of func
//Input: func is a pre-defined function
//Output: func is executed 10 times
void caller(T func){
	//Student Work:
	//First initilize a vector of threads. This will allow for easier handling
	vector<thread> myThreads;

	//For loop(int i 0 -> 10)
	// push a new thread onto the vector of thread objects
	// When a new thread is pushed, it is also initilized and started
	// be sure when initilizing pass a tid value int thread(func, i)
	for(int i = 0; i < 10; i++) {
        myThreads.push_back(thread(func, i));
    }

	//Running Threads!
	
	//For loop(int i 0-10)
	//Join all of the threads back to this main program.
    for(int i = 0; i < 10; i++) {
        myThreads[i].join();
    }

}
 

int main(){
   int y = 0; //Variable for experimenting with lambda calls
   vector<int> problem(10,0);//allocate 10 elements, initilized to 0
   
   //Start Student Code
   //Write a lambda function called lFunc
   // In the capture specification you should grab both problem and y as a reference
   // it should have an int parameter to take in the tid
   // The function should do the following:
   //		display "Hello from Thread <tid>"
   //		assign the tid to y
   //		assign tid*100 to problem[tid]
   auto lFunc = [&problem, &y] (int tid) {
        cout << "Hello from Thread " << tid << endl;
        y = tid;
        problem[tid] = tid*100;
    };

   //End Student Code
	
    caller(lFunc);

   cout << "Y is: " << y << endl;
   for(int i = 0; i < 10; i++){
       cout << "Problem["<<i<<"] = "<<problem[i]<<endl;
   } 

}