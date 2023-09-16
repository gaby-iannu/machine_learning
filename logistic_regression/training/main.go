// Package main provides ...
package main

import (
	"encoding/csv"
	"fmt"

	"github.com/gaby-iannu/df_utils"
	"github.com/gonum/matrix/mat64"
)

/*
m1 = -0.48
m2 = 1.54
*/

func main() {
	f := df_utils.OpenFile("../training_test/training.csv")
	defer f.Close()

	reader := csv.NewReader(f)
	data, err := reader.ReadAll()
	df_utils.HandlerError(err)

	featureData := make([]float64, 2*len(data))
	labels := make([]float64, len(data))

	var featureIndex int

	for i,record := range data {
		if i == 0 {
			continue
		}
		featureData[featureIndex] = df_utils.ParseFloat(record[0])
		featureIndex++
		featureData[featureIndex] = 1.0
		featureIndex++
		labels[i] = df_utils.ParseFloat(record[1])
	}

	features := mat64.NewDense(len(data), 2, featureData)
	weights := df_utils.LogisticRegression(features, labels, 100, 0.3)
	formula := "p = 1/ (1 + exp(-m1 * FICO.score - m2))"
	fmt.Printf("\n%s\n\nm1 = %0.2f\nm2 = %0.2f\n\n", formula, weights[0],weights[1])
}
