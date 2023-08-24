// Package main provides ...
package main

import (
	"encoding/csv"
	"fmt"

	"github.com/berkmancenter/ridge"
	"github.com/gaby-iannu/df_utils"
	"github.com/gonum/matrix/mat64"
)

func main() {
	f := df_utils.OpenFile("../training.csv")
	defer f.Close()

	reader := csv.NewReader(f)
	trainingData, err := reader.ReadAll()
	df_utils.HandlerError(err)

	featureData := make([]float64, 4*len(trainingData))
	yData := make([]float64, len(trainingData))

	var featureIndex int
	var yIndex int

	for i,record := range trainingData {
		if i == 0 {
			continue
		}

		for j,val := range record {
			valParsed := df_utils.ParseFloat(val)
			if j < 3 {
				if i == 0 {
					featureData[featureIndex] = 1
					featureIndex++
				}
			}
			if j == 3 {
				yData[yIndex] = valParsed
				yIndex++
			}
		}
	}

	features := mat64.NewDense(len(trainingData), 4, featureData)
	y := mat64.NewVector(len(trainingData), yData)
	fmt.Printf("features: %v\n", features)
	fmt.Printf("y: %v\n", y)
	r:= ridge.New(features, y, 1.0)
	r.XSVD = new(mat64.SVD)
	fmt.Println(r.XSVD)
	r.Regress()
	c1 := r.Coefficients.At(0,0)
	c2 := r.Coefficients.At(1,0)
	c3 := r.Coefficients.At(2,0)
	c4 := r.Coefficients.At(3,0)
	fmt.Printf("Formula: y = %0.3f + %0.3f TV + %0.3f Radio + %0.3f Newspaper\n", c1, c2, c3, c4)
}
