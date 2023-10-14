package main

import (
	"encoding/csv"
	"fmt"
	"io"
	"log"
	"os"

	"github.com/gaby-iannu/df_utils"
	"github.com/go-gota/gota/dataframe"
	"github.com/gonum/floats"
	"github.com/mash/gokmeans"
)

func calculateCentroids(f *os.File) []gokmeans.Node {
	r := csv.NewReader(f)
	var data []gokmeans.Node

	for {
		record, err := r.Read()
		if err == io.EOF {
			break
		}

		if record[0] == "Driver_ID" {
			continue
		} 

		var points []float64
		for i:=1; i<3; i++ {
			val := df_utils.ParseFloat(record[i])
			points = append(points, val)
		}

		data = append(data, gokmeans.Node{points[0], points[1]})
	}
	success, centroids := gokmeans.Train(data, 2, 50)
	if !success {
		log.Fatal("couldn't generate cluster")
	}

	return centroids
}

func calculateClusters(df dataframe.DataFrame, centroids []gokmeans.Node) ([][]float64, [][]float64) {
	var clusterOne [][]float64
	var clusterTwo [][]float64

	yVals := df.Col("Distance_Feature").Float()

	for i, xVal := range df.Col("Speeding_Feature").Float() {
		distanceOne := floats.Distance([]float64{yVals[i], xVal}, centroids[0], 2)
		distanceTwo := floats.Distance([]float64{yVals[i], xVal}, centroids[1], 2)
		if distanceOne < distanceTwo {
			clusterOne = append(clusterOne, []float64{xVal, yVals[i]})
			continue
		}
		clusterTwo = append(clusterTwo, []float64{xVal, yVals[i]})
	}
	
	return clusterOne, clusterTwo
}

func withinClusterMean(cluster [][]float64, centroids []float64) float64 {
	var meanDistance float64
	
	for _,point := range cluster {
		meanDistance += floats.Distance(point, centroids, 2) / float64(len(cluster))
	}

	return meanDistance
}

func main() {
	f := df_utils.OpenFile("../fleet_data.csv")
	defer f.Close()
	centroids := calculateCentroids(f)
	f.Seek(0, io.SeekStart)
	df := dataframe.ReadCSV(f)
	clusterOne, clusterTwo := calculateClusters(df, centroids)
	fmt.Printf("Cluster one metric: %0.2f\n", withinClusterMean(clusterOne, centroids[0]))
	fmt.Printf("Cluster two metric: %0.2f\n", withinClusterMean(clusterTwo, centroids[1]))
	// [[41.72666666666667 17.333333333333332] [59.53666666666666 24.5]]	
	// fmt.Println(centroids)
}
