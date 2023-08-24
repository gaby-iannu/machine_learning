// Package main provides ...
package main

import (
	"encoding/csv"
	"fmt"

	"github.com/gaby-iannu/df_utils"
	"github.com/sajari/regression"
)

func main() {
	f := df_utils.OpenFile("../../training.csv")
	defer f.Close()

	reader := csv.NewReader(f)
	trainingData, err := reader.ReadAll()
	df_utils.HandlerError(err)

	var r regression.Regression 

	r.SetObserved("Sales")
	r.SetVar(0, "TV")
	r.SetVar(1, "Radio")

	for i, record := range trainingData {
		if i == 0 {
			fmt.Println(record)
			continue
		}

		yVal := df_utils.ParseFloat(record[4])
		tvVal := df_utils.ParseFloat(record[1])
		radioVal := df_utils.ParseFloat(record[2])
		
		r.Train(regression.DataPoint(yVal, []float64{tvVal, radioVal}))
	}

	r.Run()
	fmt.Printf("Formula: %s\n", r.Formula)
}
