// Package main provides ...
package main

import (
	"encoding/csv"
	"fmt"
	"math"
	"os"

	"github.com/gaby-iannu/df_utils"
	"github.com/sajari/regression"
)

func main() {
	f, err	:= os.Open("../test.csv")
	df_utils.HandlerError(err)
	defer f.Close()

	reader := csv.NewReader(f)
	testData, err := reader.ReadAll()
	df_utils.HandlerError(err)

	// Loop over the test data predicting y and evaluating the prediction
	// with the mean absolute error
	var mAE float64

	// In this case we are going to try and model our Sales (y)
	// by the TV feature plus an intercept. As such, let's create
	// the struct needed to train a model usign github.com/sajari/regression
	var r regression.Regression
	r.SetObserved("Sales")
	r.SetVar(0, "TV")

	for i, record := range testData {
		if i == 0 {
			continue
		}
		
		// Parse  the observed Sales, or "y"
		yObserved := df_utils.ParseFloat(record[3])
		// Parse the TV value
		tvVal := df_utils.ParseFloat(record[0])

		// Predicted y with our trained model
		yPredicted, _ := r.Predict([]float64{tvVal})
		// df_utils.HandlerError(err)

		// Add the to the mean absolute error
		mAE += math.Abs(yObserved-yPredicted)/float64(len(testData))
	}

	fmt.Printf("MAE= %0.2f\n", mAE)
}
