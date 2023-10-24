// Package main provides ...
package main

import (
	"fmt"

	"github.com/gaby-iannu/df_utils"
)


func main() {
	
	df := df_utils.CreateDataFrame("./fleet_data.csv")
	fmt.Println(df.Describe())

	files := df_utils.CreateHistogram(df)

	fmt.Println(files)

	var  r uint8 = 255
	var  b uint8 = 128
	var  a uint8 = 255

	plotConfig := df_utils.PlotConfig{
		PathToCsv: "./fleet_data.csv",
		YColName: "Distance_Feature",
		XColName: "Speeding_Feature",
		XLabel: "Speeding",
		YLabel: "Distance",
		PloterFile: "fleet_data_scatter.png",
		VgPoints: 3,
		R: &r,
		B: &b,
		A: &a,
		Style: "Glyph",
	}
	df_utils.CreateScatterPlot(plotConfig)
}
