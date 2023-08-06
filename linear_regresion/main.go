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

func handlerError(err error) {
	if err != nil {
		panic(err)
	}
}
func createPlot(col string) *plot.Plot {
	// Create the plot
	p := plot.New()
	p.X.Label.Text = col
	p.Y.Label.Text = "y"
	p.Add(plotter.NewGrid())
	return p
}


func doScatterPlots(df dataframe.DataFrame) {
	// Extract the target column.
	yVals := df.Col("Sales").Float()
	
	// Create a scatter plot for each of the features in the datasets
	for _,col := range df.Names() {
		// pts will hold the values for plotting
		pts := make(plotter.XYs, df.Nrow())

		// Fill pts with data
		for i,fVal := range df.Col(col).Float() {
			pts[i].X = fVal
			pts[i].Y = yVals[i]
		}
		p := createPlot(col)

		s,err := plotter.NewScatter(pts)
		handlerError(err)
		s.GlyphStyle.Radius = vg.Points(3)
		// Save the plot to PNG file.
		p.Add(s)
		err = p.Save(4*vg.Inch, 4*vg.Inch, col + "_scatter.png")
		handlerError(err)
	}
}

func main() {
	f, err 	:= os.Open("./Advertising.csv")
	handlerError(err)
	defer f.Close()

	// Create dataframe from csv file.
	df := dataframe.ReadCSV(f)

	// Use describe method to calculate summary statistics
	// for all of the columns in one shot.
	summary := df.Describe()
	fmt.Println(summary)
	for _, col := range df.Names() {
		// Create a plotter.Values value and fill it with the values
		// from the respective column of the dataframe.
		plotValues := make(plotter.Values, df.Nrow())
		for i,fVal := range df.Col(col).Float() {
			plotValues[i] = fVal
		}

		// Make a plot and set its title
		p  := plot.New()
		p.Title.Text = fmt.Sprintf("Histogram of %s", col)

		// Create a histogram of our values drawn from the standard normal
		h, err := plotter.NewHist(plotValues, 16)
		handlerError(err)

		// Normalize the histogram
		h.Normalize(1)
		// Add the histogram to the plot
		p.Add(h)

		// Save the plot to a PNG file
		if err = p.Save(4*vg.Inch, 4*vg.Inch, col + "_hist.png"); err != nil {
			panic(err)
		}
	}

	doScatterPlots(df)
}
