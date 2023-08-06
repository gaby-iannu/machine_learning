// Package main provides ...
package main

import (
	"encoding/csv"
	"fmt"
	"io"
	"math"
	"os"
	"strconv"

	"gonum.org/v1/gonum/stat"
)

func handleError (err error) {
	if err != nil {
		panic(err)
	}
}

func main() {
	f,err := os.Open("../continuous_data.csv")
	handleError(err)
	defer f.Close()
	// Create a new CSV reader reading from open file
	reader := csv.NewReader(f)

	// observed and predicted will hold the parsed observed and predicted values
	// form the continuos data file 
	var observed []float64
	var predicted []float64
	
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

		// Read in the observed and predicted values.
		observedVal, err := strconv.ParseFloat(record[0], 64)
		handleError(err)
		observed = append(observed, observedVal)


		predictedVal, err := strconv.ParseFloat(record[1], 64)
		handleError(err)
		predicted = append(predicted, predictedVal)
	}

	// Calculate the mean absolute error and mean squared error.
	var mAE float64
	var mSE float64
	for i, oVal := range observed {
		mAE += math.Abs(oVal - predicted[i]) / float64(len(observed))
		mSE += math.Pow(oVal - predicted[i], 2) / float64(len(observed))
	}

	meanObserved := stat.Mean(observed, nil)
	percentageMean := (mAE*100)/meanObserved

	// Calculate R-squared value
	rSquared := stat.RSquaredFrom(observed, predicted, nil)

	fmt.Printf("Mean Absolute Error: %0.2f\n", mAE)
	fmt.Printf("Mean Squared Error: %0.2f\n", mSE)
	fmt.Printf("Mean Observed: %0.2f\n", meanObserved)
	fmt.Printf("Percentage of Mean: %0.2f%%\n", percentageMean)
	fmt.Printf("R-Squared: %0.2f\n", rSquared)

}
