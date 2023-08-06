// Package main provides ...
package main

import (
	"log"
	"os"

	"github.com/go-gota/gota/dataframe"
	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/vg"
)

func main() {
	
	// Open CSV file
	irisFile, err := os.Open("../iris.csv")
	if err != nil {
		panic(err)
	}
	defer irisFile.Close()
	// Create dataframe form csv file
	irisDF := dataframe.ReadCSV(irisFile)
	
	// Create a plot and set its title and axis label
	p := plot.New()
	p.Title.Text = "Box plots"
	p.Y.Label.Text = "Values"
	
	// Create the box for our data
	w := vg.Points(50)
	for idx, colName := range irisDF.Names() {
		// If the column is one of the feature column, let's create 
		// a histogram of the values
		if colName != "species" && colName != "variety" {
			// Create a plotter.Values value and fill it with the values
			// from the respective column of the dataframe
			v := make(plotter.Values, irisDF.Nrow())
			for i, floatV := range irisDF.Col(colName).Float() {
				v[i] = floatV
			}

			// Add the data to the plot
			b, err := plotter.NewBoxPlot(w, float64(idx), v)
			if err != nil {
				log.Println(colName)
				log.Println(v)
				panic(err)
			
			p.Add(b)
		}
	}
	// Set the X axis of the plot to nominal with
	// the given names for x=0, x=1, etc 
	p.NominalX("sepal_length", "sepal_width", "petal_length", "petal_width")
	if err = p.Save(6*vg.Inch, 8*vg.Inch, "boxplot.png"); err != nil {
		panic(err)
	} 
}
