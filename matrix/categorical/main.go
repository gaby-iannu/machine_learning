// Package main provides ...
package main

import (
	"encoding/csv"
	"fmt"
	"io"
	"os"
	"strconv"
)


func handleError(err error) {
	if err != nil {
		panic(err)
	}
}

func addValue(v string, a []int) []int {
	iVal, err := strconv.Atoi(v) 
	handleError(err)
	a = append(a, iVal)
	return a
}

func main() {
	var observed []int
	var predicted []int

	f, err := os.Open("../labeled.csv")
	handleError(err)
	defer f.Close()

	reader := csv.NewReader(f)
	skipHead := true
	for {
		record, err := reader.Read()
		if err == io.EOF {
			break
		}

		if skipHead {
			skipHead = false
			continue
		}
		observed = addValue(record[0], observed)
		predicted = addValue(record[1], predicted)
	}

	// This variable will hold our count of true positive and true negative values
	var truePosNeg int

	// Accumulate the true positive/negative count.
	for i, oVal := range observed {
		if oVal == predicted[i] {
			truePosNeg++
		}
	}
	// Calculate accuracy value standard out.
	accuracy := float64(truePosNeg)/float64(len(observed))
	fmt.Printf("Accurace: %0.2f\n", accuracy)

	classes := []int{1, 2, 3}
	for i,class := range classes {
		var truePos int
		var falsePos int
		var falseNeg int

		for _, oVal := range observed {
			switch oVal {
			// If the observed value is the relevant class, we should check to see if we predicted that class.
			case class:
				if predicted[i] == class {
					truePos++
					continue
				}
				falseNeg++
			// If the observed value is a different class, we should check to see if we predicted a false positive
			default:
				if predicted[i] == class {
					falsePos++
				}
			}
		}
		// Calculate the precision
		precision := float64(truePos) / float64(truePos + falsePos)
		// Calculate the recall
		recall := float64(truePos) / float64(truePos + falseNeg)

		fmt.Printf("Precision (class %d): %0.5f\n", class, precision)
		fmt.Printf("Recall (class %d): %0.2f\n", class, recall)
	}
}
