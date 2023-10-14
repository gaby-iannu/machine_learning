// Package main provides ...
package main

import (
	"fmt"

	"github.com/gaby-iannu/df_utils"
	"github.com/go-gota/gota/dataframe"
)

type centroid []float64

func main() {
	df := df_utils.CreateDataFrame("../iris.csv")
	speciesNames := []string{
		"Iris-setosa",
		"Iris-versicolor",
		"Iris-virginica",
	}

	centroids := make(map[string]centroid)

	for _,specie := range speciesNames {
		filter := dataframe.F{
			Colname: "species",
			Comparator: "==",
			Comparando: specie,
		}
		filtered := df.Filter(filter)

		summary := filtered.Describe()
		var c centroid
		for _,feature := range summary.Names() {
			if feature == "column" || feature == "species" {
				continue
			}
			c = append(c, summary.Col(feature).Float()[0])
		}
		centroids[specie] = c
	}

	for _,specie := range speciesNames {
		fmt.Printf("%s centroid: %v\n", specie,centroids[specie])
	}
}
