// Package main provides ...
package main

import (
	"fmt"
	"os"

	"github.com/go-gota/gota/dataframe"
	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/vg"
)


func main() {
	
	irisFile, err := os.Open("../iris.csv")
	if err != nil {
		panic(err)
	}

	defer irisFile.Close()
	
	irisDF := dataframe.ReadCSV(irisFile)
	for _,colName := range irisDF.Names() {
		// if column is one of the feature column, let's create 
		// histogram of the values
		if colName != "species" {
			// Create a plotter.Values value and fill it with the values
			// from the respective column of the dataframe 
			v := make(plotter.Values, irisDF.Nrow())
			for i, floatV := range irisDF.Col(colName).Float() {
				v[i] = floatV
			}

			// Make a plot and set its title
			p := plot.New()
			p.Title.Text = fmt.Sprintf("Histogram of a %s", colName)

			// Create a histogram of our values drawn
			// from the standard normal.
			h, err := plotter.NewHist(v, 16)
			if err != nil {
				panic(err)
			}
			// Normalize the histogram
			h.Normalize(1)
			// Add to the histogram to the plot
			p.Add(h)
			// Save the plot to a PNG file
			if err = p.Save(4*vg.Inch, 4*vg.Inch, colName+"_hist.png"); err != nil {
				panic(err)
			}
		}
	}
}
