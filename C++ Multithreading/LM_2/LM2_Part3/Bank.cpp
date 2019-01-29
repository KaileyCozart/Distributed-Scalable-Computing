// LM2 - Threads and Resource Contention
// CS315 - Whitworth University
// Kailey Cozart

#include <iostream>
#include <vector>
#include <thread>
#include <mutex>
using namespace std;

class BankAccount {
private:
    int accountId = 0;
    float balance = 0;
    mutex gu;
    bool priorityActive = false;
public:
    BankAccount(int accountId, float balance);
    void PriorityWithdrawal(float amount);
    void Withdrawal(float amount);
    void Deposit(float amount);
    void Print(int accountId, float balance);
};

BankAccount::BankAccount(int accountId, float balance) {
    this->accountId = accountId;
    this->balance = balance;
}

void BankAccount::PriorityWithdrawal(float amount) {
    while (1) {
        priorityActive = true;
        gu.lock();
        if ((balance - amount) > 0) {
            balance = balance - amount;
            cout << "PRIORITY WITHDRAWAL has occured." << endl;
            Print(accountId, balance);
            cout << " " << endl;
            gu.unlock();
            priorityActive = false;
            return;
        }
        else {
            gu.unlock();
            // delay
            this_thread::sleep_for (chrono::milliseconds(1));
        }
    }
}

void BankAccount::Withdrawal(float amount) {
    while (1) {
        gu.lock();
        if ((balance - amount) > 0 && priorityActive == false) {
            balance = balance - amount;
            cout << "WITHDRAWAL has occured." << endl;
            Print(accountId, balance);
            cout << " " << endl;
            gu.unlock();
            return;
        }
        else {
            gu.unlock();
            // delay
            this_thread::sleep_for (chrono::milliseconds(1));
        }
    }
}

void BankAccount::Deposit(float amount) {
    gu.lock();
    balance = balance + amount;
    cout << "DEPOSIT has occured." << endl;
    Print(accountId, balance);
    cout << " " << endl;
    gu.unlock();
    // delay
    this_thread::sleep_for (chrono::milliseconds(1));
}

void BankAccount::Print(int accountId, float balance) {
    cout << "The current balance for account" << accountId << " is $" << balance << endl;
}

int main(){
    
    BankAccount * createdAccount = new BankAccount(1, 0.0);
    
    auto lFunc1 = [&createdAccount] (int tid) {
        for(int i = 0; i < 30; i ++) {
            createdAccount->PriorityWithdrawal(15.0);
        }
    };
    thread td1 = thread(lFunc1, 2);
    
    auto lFunc2 = [&createdAccount] (int tid) {
        for(int i = 0; i < 30; i ++) {
            createdAccount->Withdrawal(13.0);
        }
    };
    thread td2 = thread(lFunc2, 0);
    
    auto lFunc3 = [&createdAccount] (int tid) {
        for(int i = 0; i < 330; i ++) {
            createdAccount->Deposit(4.0);
        }
    };
    thread td3 = thread(lFunc3, 1);
    
    td1.join();
    td2.join();
    td3.join();
    
    return 0;
}
