// Package main provides ...
package main

import (
	"encoding/csv"
	"fmt"
	"os"

	"github.com/gaby-iannu/df_utils"
	"github.com/sajari/regression"
)


func main() {
	f, err := os.Open("../training.csv")
	df_utils.HandlerError(err)
	defer f.Close()

	// Create a new csv reader reading from the opened file
	reader := csv.NewReader(f)

	// Read in all of the csv records
	// reader.FieldsPerRecord = 4
	trainingData, err := reader.ReadAll()
	df_utils.HandlerError(err)

	// In this case we are going to try and model our Sales (y)
	// by the TV feature plus an intercept. As such, let's create
	// the struct needed to train a model usign github.com/sajari/regression
	var r regression.Regression
	r.SetObserved("Sales")
	r.SetVar(0, "TV")

	// Loop of records in the CSV, adding the training data
	// to the regression value
	skip := true
	for _, record := range trainingData {
		// Skip the header
		if skip {
			skip = false
			continue
		}

		// Parse the Sales regression measure, or "y" 
		yVal := df_utils.ParseFloat(record[3])

		// Parse the TV value
		tvVal := df_utils.ParseFloat(record[0])

		// Add these point to the regression value
		r.Train(regression.DataPoint(yVal, []float64{tvVal}))
	}

	r.Run()
	fmt.Printf("Regression Formula: %s\n", r.Formula)
}
