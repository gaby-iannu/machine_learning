// Package main provides ...
package main

import (
	"fmt"

	"github.com/gaby-iannu/df_utils"
	"github.com/go-gota/gota/dataframe"
)

func main() {
	path := "./Advertising.csv"
	df := df_utils.CreateDataFrame(path)
	df_training, df_test := df_utils.CreateTrainingAndTestSet(df, 4, 5)
	
	fmt.Println(df_training.Describe())
	fmt.Println(df_test.Describe())

	maps := map[int]dataframe.DataFrame{
		0: df_training,
		1: df_test,
	}

	for i,path := range []string{"../training.csv","../test.csv"} {
		df_utils.SaveDataFrameToCSV(path, maps[i])
	}
}
