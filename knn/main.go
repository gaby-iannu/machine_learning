package main

import (
	"fmt"
	"math"

	"github.com/gaby-iannu/df_utils"
	"github.com/sjwhitworth/golearn/base"
	"github.com/sjwhitworth/golearn/evaluation"
	"github.com/sjwhitworth/golearn/knn"
)

func main() {

	irisData, err := base.ParseCSVToInstances("./iris.csv", true)
	df_utils.HandlerError(err)

	knn := knn.NewKnnClassifier("euclidean", "linear", 2)
	cv, err := evaluation.GenerateCrossFoldValidationConfusionMatrices(irisData, knn, 5)
	df_utils.HandlerError(err)

	mean, variance := evaluation.GetCrossValidatedMetric(cv,evaluation.GetAccuracy)
   	stdev := math.Sqrt(variance)

	// Output the cross metrics to standard out.
   	fmt.Printf("\nAccuracy\n%.2f (+/- %.2f)\n\n", mean, stdev*2)
}
