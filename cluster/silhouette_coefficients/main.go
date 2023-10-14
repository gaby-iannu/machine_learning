package main

import (
	"fmt"

	"github.com/gaby-iannu/df_utils"
	"github.com/go-gota/gota/dataframe"
	"github.com/gonum/floats"
)

func dfFloatRow(df dataframe.DataFrame, names []string, idx int) []float64 {
	var row []float64
	for _, name := range names {
		row = append (row, df.Col(name).Float()[idx])
	}
	return row
}

func dfFiltered(df dataframe.DataFrame) map[string]dataframe.DataFrame {
	cluster := make(map[string]dataframe.DataFrame)

	filter := dataframe.F{
		Colname: "species",
		Comparator: "==",
	}

	speciesNames := []string{
		"Iris-setosa",
		"Iris-versicolor",
		"Iris-virginica",
	}

	for _,name := range speciesNames {
		filter.Comparando = name
		cluster[name] = df.Filter(filter)
	}

	return cluster
}

type centroid []float64

func calculateCentroid(cluster map[string]dataframe.DataFrame) map[string]centroid {


	centroids := make(map[string]centroid)

	for specie,filtered := range cluster {

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

	return centroids

}

func main() {
	df := df_utils.CreateDataFrame("../iris.csv")
	cluster	:= dfFiltered(df)

	centroids := calculateCentroid(cluster)

	labels := df.Col("species").Records()
	floatColumns := []string{
		"sepal_length",
		"sepal_width",
		"petal_length",
		"petal_width",
	}

	var silhouette_coefficients float64

	speciesNames := []string{
		"Iris-setosa",
		"Iris-versicolor",
		"Iris-virginica",
	}

	for idx,label := range labels {
		var a float64
		for i:=0; i<cluster[label].Nrow(); i++ {
			current := dfFloatRow(df, floatColumns, idx)
			other := dfFloatRow(cluster[label], floatColumns, i)
			a += floats.Distance(current, other, 2) / float64(cluster[label].Nrow())
		}

		var otherCluster string
		var distanceToCluster float64
		for _, specie := range speciesNames {
			if specie == label {
				continue
			}
			distanceForThisCluster := floats.Distance(centroids[label], centroids[specie], 2)
			if distanceToCluster == 0.0 || distanceForThisCluster < distanceToCluster {
				otherCluster = specie
				distanceToCluster = distanceForThisCluster
			}
		}

		var b float64
		for i:=0; i<cluster[otherCluster].Nrow(); i++ {
			current := dfFloatRow(df, floatColumns, idx)
			other := dfFloatRow(cluster[otherCluster], floatColumns, i)
			b += floats.Distance(current, other, 2) / float64(cluster[otherCluster].Nrow())

		}
		if a > b {
			silhouette_coefficients += ((b-a)/a) / float64(len(labels))
		}
		silhouette_coefficients += ((b-a)/b) / float64(len(labels)) 
	}

	fmt.Printf("Average silhouette coefficient: %0.2f\n", silhouette_coefficients)
}
