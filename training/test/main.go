// Package main provides ...
package main

import (
	"bufio"
	"os"

	"github.com/go-gota/gota/dataframe"
)

func handleError(err error) {
	if err != nil {
		panic(err)
	}
}

func createSubset(lenght, additional int) []int {
	v := make([]int, lenght)
	for i:=0; i<lenght; i++ {
		v[i] = i + additional
	}
	return v
}

func calculateElementEachSet(df dataframe.DataFrame) (int, int) {
	trainingNum := (4 * df.Nrow()) / 5
	testNum := df.Nrow() / 5
	if trainingNum + testNum < df.Nrow() {
		trainingNum++
	}
	return trainingNum, testNum
}

func main() {
	f,err := os.Open("../diabetes.csv")
	handleError(err)
	defer f.Close()

	// Create dataframe from csv file
	// types of columns will be inferred
	diabetesDF := dataframe.ReadCSV(f)

	// Calculate the number of element each set
	trainingNum, testNum := calculateElementEachSet(diabetesDF)
	// Create the subset indices
	trainingIdx := createSubset(trainingNum, 0) 
	testIdx := createSubset(testNum, trainingNum) 

	// Create the subset dataframes
	trainingDF := diabetesDF.Subset(trainingIdx)
	testDf := diabetesDF.Subset(testIdx)

	// Create a map that will be used in writing the data to files
	setMap := map[int]dataframe.DataFrame {
		0: trainingDF,
		1: testDf,
	}

	// Create respective files
	for idx, name := range []string {"training.csv","test.csv"} {
		// Save the filtered dataset file 
		f, err := os.Create(name)
		handleError(err)
		
		// Create a buffered writer
		w := bufio.NewWriter(f)
		err = setMap[idx].WriteCSV(w)
		handleError(err)

	}
}
