package main

import "fmt"
import "math/rand"
import "time"

func main() {
	fmt.Printf("\n")
	fmt.Println("WELCOME TO SUDOKU")
	fmt.Println("Enter the number of clues you would like: ")
    var input int
    fmt.Scanln(&input)
	fmt.Println("GENERATING...")
	generateGrid(9, input)
	
}

func generateGrid(n int, clues int) {


	grid := make([][]int, n)
	failGrid := make([][]int, n)
	puzzle := make([][]int, n)
	for i := range grid {
		grid[i] = make([]int, n)
		failGrid[i] = make([]int, n)
		puzzle[i] = make([]int, n)
	}
	for a:= 0; a < n; a++ {
		for b:=0; b < n; b++ {
			grid[a][b] = 0
			failGrid[a][b] = 0
			puzzle[a][b] = 0
		}
	}

	for a := 0; a < n; a++ {
		for b := 0; b < n; b++ {

			column := make([]int, n)
			for i := 0; i < n; i++ {
				column[i] = grid[i][b] 
			}

			row := make([]int, n)
			for i := 0; i < n; i++ {
				row[i] = grid[a][i]
			}
			
			square := make([]int, 0)
			var sizeSquare int = n / 3
			var curRow int = a % sizeSquare
			var curColumn int = b % sizeSquare
			var startRow int = a - curRow
			var startColumn int = b - curColumn

			var maxSize1 int = startRow + sizeSquare
			var maxSize2 int = startColumn + sizeSquare
			for i := startRow; i < maxSize1; i++ {
				for j := startColumn; j < maxSize2; j++ {
					square = append(square, grid[i][j])
				}
			}
			combined := make([]int, 0)
			combined = append(combined, column...)
			combined = append(combined, row...)
			combined = append(combined, square...)

			var result int = checkValues(combined, n)

			if result == 0  {
				failGrid[a][b]++
				var backStep = failGrid[a][b]
				
				if failGrid[a][b] >= a*b {
					failGrid[a][b] = 0
				}

				for i := 0; i < backStep+1; i++ {
					grid[a][b] = 0
					if b==0 {
						b = n - 2
						if a != 0 {
							a--
						}
					} else {
						b = b - 1
					}
				}			
				b--
	
			} else {
				grid[a][b] = result
			}
			

		}
	}

	if clues > n*n {
		clues = n*n
	}
	for i := 0; i < clues ;i++ {
		rand.Seed(time.Now().UnixNano())
		var x int = (rand.Intn(n))
		rand.Seed(time.Now().UnixNano())
		var y int = (rand.Intn(n))

		if puzzle[x][y] == 0 {
			puzzle[x][y] = grid[x][y]
		} else {
			i--
		}
	}

	fmt.Println("GRID:")
	for i := 0; i < len(grid); i++ {
		fmt.Println(grid[i])
	}
	fmt.Println("PUZZLE:")
	for i := 0; i < len(puzzle); i++ {
		fmt.Println(puzzle[i])
	}
}

func checkValues(arr []int, n int) int {
	boolArray := make([]bool, n)
	for i := 0; i < len(boolArray); i++ {
		boolArray[i] = false
	}

	for i := 0; i < len(arr); i++ {
		if arr[i] != 0 {
			boolArray[arr[i] - 1] = true
		}
	}
	possibleValues := make([]int, 0)
	for i := 0; i < len(boolArray); i++ {
		if boolArray[i] == false {
			possibleValues = append(possibleValues, i+1)
		}
	}

	if (len(possibleValues) == 0) {
		return 0
	} else {
		rand.Seed(time.Now().UnixNano())
		var ranVal int = (rand.Intn(len(possibleValues)))
		var value int = possibleValues[ranVal]

		return value
	}
	
}
