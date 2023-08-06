// Package main provides ...
package main

import (
	"fmt"

	"gonum.org/v1/gonum/stat"
	"gonum.org/v1/gonum/stat/distuv"
)


func main() {
	// Define the observed frequencies
	observed := []float64{
		260.0, // This number is the number of observed with no regular exercise.
		135.0, // This number is the number of observed with sporatic exercise.
		105.0,  // This number is the number of observed with regular exercise.
	}

	// Define the total observed
	totalObserved := 500.0

	// Calculate the expected frequencies (again assuming the null Hypothesis)
	expected := []float64{
		totalObserved * 0.60,
		totalObserved * 0.25,
		totalObserved * 0.15,
	}

	// Calculate the ChiSquare test statisc
	chisquare := stat.ChiSquare(observed, expected)
	fmt.Printf("Chi Square: %0.2f\n", chisquare)

	// Create a chi-squared distribution with k degrees of freedom.
	// In this case we have k=3-1=2, because degree of freedom 
	// for a chi-squared distribution is the number of posible 
	// categories minus one. 
	chDist := distuv.ChiSquared{
		K: 2.0,
		Src: nil,
	}

	// Calculate the p-values for our specific test statistic.
	pValue := chDist.Prob(chisquare)
	fmt.Printf("p-value: %0.4f\n", pValue)
}
