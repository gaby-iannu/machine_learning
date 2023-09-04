// Package main provides ...
package main

import (
	"github.com/gaby-iannu/df_utils"
)

func main() {
	 df := df_utils.CreateDataFrame("../clean_data/clean_loan_data.csv")

	 dtr, dte := df_utils.CreateTrainingAndTestSet(df, 4, 5)
	 df_utils.SaveDataFrameToCSV("training.csv", dtr)
	 df_utils.SaveDataFrameToCSV("test.csv", dte)
}
