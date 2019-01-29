package main

import "fmt"
import "time"
import "math/rand"

//////////////////////////
// Brand Information
//////////////////////////

type Brand int
const (
	Coke Brand = 0
	Pepsi Brand = 1
)

///////////////////////////
// Product and Cost Holder
///////////////////////////

type inventory struct {
	inventoryCokeCount int
	inventoryPepsiCount int
	shelfCokeCount int
	shelfPepsiCount int
	checkoutCokeCount int
	checkoutPepsiCount int
	costCoke float32
	costPepsi float32
	checkoutLedger float32
}

///////////////////////////
// Return Cost Functions
///////////////////////////

func (myInv *inventory)GetCokeCost() (cokeCost float32){
	cokeCost = myInv.costCoke
	return
}

func (myInv *inventory)GetPepsiCost() (pepsiCost float32){
	pepsiCost = myInv.costPepsi
	return
}

func (myInv *inventory)GetCheckoutEarnings() (checkoutEarnings float32){
	checkoutEarnings = myInv.checkoutLedger
	return
}

//////////////////////////
// Coke Delivery Person
//////////////////////////

// Async Go Routine
// Deliver 480 cans (for loop)
// Deliver 24 cans to stocker and output
// Add $6 to charge for coke and output
// After every delivery, sleep for .5 second

func (storeShelf *inventory)CokeDelivery(){
	for i := 0; i < 20; i++ {
		storeShelf.inventoryCokeCount +=24
		storeShelf.costCoke += 6.0
		time.Sleep(5000 * time.Millisecond)
		fmt.Println("24 cans of Coke added to the shelf by the stocker")
	}
}

//////////////////////////
// Pepsi Delivery Person
//////////////////////////

// Async Go Routine
// Deliver 480 cans (for loop)
// Deliver 24 cans to stocker and output
// Add $4.80 to charge for coke and output
// After every delivery, sleep for .5 seconds

func (storeShelf *inventory)PepsiDelivery(){
	for i := 0; i < 20; i++ {
		storeShelf.inventoryPepsiCount += 24
		storeShelf.costPepsi += 4.8
		time.Sleep(5000 * time.Millisecond)
		fmt.Println("24 cans of Pepsi added to the shelf by the stocker")
	}
}

//////////////////////////
// Shelf Stocking Person
//////////////////////////

// Async Go Routine
// (while loop)
	// Randomly choose coke or pepsi
	// Sleep for 10 
	// place one on shelf and output

func (thisInv *inventory)StockShelf() { // Decrement inventory where getting it from it 
	for (thisInv.inventoryCokeCount >= 0 || thisInv.inventoryPepsiCount >= 0) {
		time.Sleep(10 * time.Millisecond)
		rand.Seed(time.Now().UnixNano())
		var brandSelector int = (rand.Intn(2)) + 1
		if (brandSelector == 1) {
			thisInv.shelfCokeCount += 1
			fmt.Println("1 Coke Can Placed on Shelf")
		} else if (brandSelector == 2) {
			thisInv.shelfPepsiCount += 1
			fmt.Println("1 Pepsi Can Placed on Shelf")
		}
	}
}

//////////////////////////
// Multiple Customers
//////////////////////////

// Async Go Routine
// if either coke or pepsi counter <= 30 spawn new customer
	// wait 75 
// (while loop)
	// randomly select coke or pepsi
	// increment coke or pepsi counter based on selection
	// randomly select 6, 12, 18, 24 cans to buy
	// check with stocker (for loop)
		// if not enough available, rest 100  (or whatever amount of time wanted)
		// when available, add to checkout count for pepsi or coke and output

func (thisShelf *inventory)MakeCustomers() {
	for (thisShelf.shelfCokeCount >= 0 || thisShelf.shelfPepsiCount >= 0) {
		time.Sleep(75 * time.Millisecond)
		rand.Seed(time.Now().UnixNano())
		var brandSelector int = (rand.Intn(2)) + 1
		if (brandSelector == 1) {
			time.Sleep(75 * time.Millisecond)
			rand.Seed(time.Now().UnixNano())
			var canNumSelector int = (rand.Intn(4)) + 1
			var actualNumCoke int
			if (canNumSelector == 1) {
				actualNumCoke = 6
			} else if (canNumSelector == 2) {
				actualNumCoke = 12
			} else if (canNumSelector == 3) {
				actualNumCoke = 18
			} else {
				actualNumCoke = 24
			}
			var numRemovedCoke int = 0
			for (numRemovedCoke == 0) {
				if (thisShelf.shelfCokeCount > actualNumCoke) {
					thisShelf.shelfCokeCount = thisShelf.shelfCokeCount - actualNumCoke
					numRemovedCoke = actualNumCoke
				}
				time.Sleep(10 * time.Millisecond)
			}
			thisShelf.checkoutCokeCount = thisShelf.checkoutCokeCount + numRemovedCoke
			fmt.Println(numRemovedCoke, " Cans of Coke Removed from Shelf by Customer")
		} else if (brandSelector == 2) {
			time.Sleep(75 * time.Millisecond)
			rand.Seed(time.Now().UnixNano())
			var canNumSelector2 int = (rand.Intn(4)) + 1
			var actualNumPepsi int
			if (canNumSelector2 == 1) {
				actualNumPepsi = 6
			} else if (canNumSelector2 == 2) {
				actualNumPepsi = 12
			} else if (canNumSelector2 == 3) {
				actualNumPepsi = 18
			} else {
				actualNumPepsi = 24
			}
			var numRemovedPepsi int = 0
			for (numRemovedPepsi == 0) {
				if (thisShelf.shelfPepsiCount > actualNumPepsi) {
					thisShelf.shelfPepsiCount = thisShelf.shelfPepsiCount - actualNumPepsi
					numRemovedPepsi = actualNumPepsi
				}
				time.Sleep(10 * time.Millisecond)
			}
			thisShelf.checkoutPepsiCount = thisShelf.checkoutPepsiCount + numRemovedPepsi
			fmt.Println(numRemovedPepsi, " Cans of Pepsi Removed from Shelf by Customer")
		}
	}
}

//////////////////////////
// Checkout Person
//////////////////////////

// Async Go Routine
// (while loop)
	// if a coke can
		// add .55 to store earning sum and output
		// decrement coke
		// sleep 10 
	// if a pepsi can
		// add .50 to store earning sum and output
		// decrement pepsi
		// sleep 10 

func (thisCheckout *inventory)CheckoutItems(){
	for {
		if (thisCheckout.checkoutCokeCount >= 0 || thisCheckout.checkoutPepsiCount >= 0) {
			time.Sleep(10 * time.Millisecond)
			rand.Seed(time.Now().UnixNano())
			var brandSelector int = (rand.Intn(2)) + 1
			if (brandSelector == 1) {
				thisCheckout.checkoutCokeCount = thisCheckout.checkoutCokeCount - 1
				thisCheckout.checkoutLedger += .55
				fmt.Println("1 Coke Can Sold at Checkout")
			} else if (brandSelector == 2) {
				thisCheckout.checkoutPepsiCount = thisCheckout.checkoutPepsiCount - 1
				thisCheckout.checkoutLedger += .50
				fmt.Println("1 Pepsi Can Sold at Checkout")
			}
		} else {
			time.Sleep(10 * time.Millisecond)
		}
	}
}

//////////////////////////
// Main Function
//////////////////////////

func main(){

	mainShelf := inventory{0,0,0,0,0,0,0.0,0.0,0.0}

	go mainShelf.CokeDelivery()
	go mainShelf.PepsiDelivery()
	go mainShelf.StockShelf()
	go mainShelf.MakeCustomers()
	go mainShelf.CheckoutItems()

	time.Sleep(60 * time.Second)

	//////////////////////////
	// Reconcile Bill
	//////////////////////////

	var cokeCost float32 = mainShelf.GetCokeCost()
	var pepsiCost float32 = mainShelf.GetPepsiCost()
	var earnings float32 = mainShelf.GetCheckoutEarnings()

	var totalCharges float32 = cokeCost + pepsiCost
	var totalProfit float32 = earnings - totalCharges
	fmt.Println("The total checkout earnings were $", earnings)
	fmt.Println("The total cost of stocking was $", totalCharges)
	fmt.Println("The total profit was $", totalProfit)

}