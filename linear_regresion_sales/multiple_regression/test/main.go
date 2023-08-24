// Package main provides ...
package main

import (
	"encoding/csv"
	"fmt"
	"math"

	"github.com/gaby-iannu/df_utils"
	"github.com/sajari/regression"
)

func main() {
	f := df_utils.OpenFile("../../test.csv")	
	defer f.Close()

	reader := csv.NewReader(f)
	trainingTest, err := reader.ReadAll()
	df_utils.HandlerError(err)

	var mAE float64
	var r regression.Regression

	for i,record := range trainingTest {
		if i == 0 {
			continue
		}
		
		yObserved := df_utils.ParseFloat(record[4])
		tvVal := df_utils.ParseFloat(record[1])
		radioVal := df_utils.ParseFloat(record[2])
		yPredicted,_ := r.Predict([]float64{tvVal, radioVal}) 
		// df_utils.HandlerError(err)
		mAE += math.Abs(yObserved-yPredicted)/float64(len(trainingTest))
	}
	fmt.Printf("MAE: %0.2f\n", mAE)
}
