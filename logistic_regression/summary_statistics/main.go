// Package main provides ...
package main

import (
	"fmt"

	"github.com/gaby-iannu/df_utils"
)


func main() {
	loanDF := df_utils.CreateDataFrame("../clean_data/clean_loan_data.csv")

	fmt.Println(loanDF.Describe())
	
	for _, file := range df_utils.CreateHistogram(loanDF) {
		fmt.Printf("Histogram created: %s\n", file)
	}
}
