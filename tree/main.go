// Pacage main provides ...
package main

import (
	"fmt"
	"math"
	"math/rand"

	"github.com/gaby-iannu/df_utils"
	"github.com/sjwhitworth/golearn/base"
	"github.com/sjwhitworth/golearn/evaluation"
	"github.com/sjwhitworth/golearn/trees"
)


func main() {
	
	irisData, err := base.ParseCSVToInstances("./iris.csv", true)
	df_utils.HandlerError(err)


	rand.Seed(44111342)

	tree := trees.NewID3DecisionTree(0.6)
	cv, err := evaluation.GenerateCrossFoldValidationConfusionMatrices(irisData, tree, 5)
	df_utils.HandlerError(err)

	// Get the mean, variance and standard desviation of the accuracy for the 
	// cross validation
	mean, variance := evaluation.GetCrossValidatedMetric(cv, evaluation.GetAccuracy)
	stdev := math.Sqrt(variance)
	fmt.Printf("Accuracy: %.2f (+/- %.2f)\n", mean, stdev*2)
}
