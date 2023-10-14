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

	df_utils.CreateScatterPlot("./fleet_data.csv", "Distance_Feature", "Speeding_Feature", "Speeding",
	 "Distance", "fleet_data_scatter.png")
}
