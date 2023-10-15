package main

import (
	"fmt"

	"github.com/gaby-iannu/df_utils"
)

const path_to_csv string = "./AirPassengers.csv"

func printDF() {
	df := df_utils.CreateDataFrame(path_to_csv)
	fmt.Println(df)
}

func doPlot() {
	var  b uint8 = 255
	var  a uint8 = 255

	plotConfig := df_utils.PlotConfig{
		PathToCsv: path_to_csv,
		YColName: "value",
		XColName: "time",
		XLabel: "time",
		YLabel: "passengers",
		PloterFile: "passengers_ts.png",
		VgPoints: 1,
		B: &b,
		A: &a,
		Style: "Line",
	}

	df_utils.CreateScatterPlot(plotConfig)
}

func main() {
	printDF()
	doPlot()
}
