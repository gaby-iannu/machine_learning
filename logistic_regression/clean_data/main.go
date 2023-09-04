// Package main provides ...
package main

import (
	"encoding/csv"
	"os"
	"strconv"
	"strings"

	"github.com/gaby-iannu/df_utils"
)

const (
	 scoreMax = 830.0
	 scoreMin = 640.0
)

func main() {
	f := df_utils.OpenFile("../loan_data.csv")
	defer f.Close()

	reader := csv.NewReader(f)
	data , err := reader.ReadAll()
	df_utils.HandlerError(err)

	cf, err := os.Create("clean_loan_data.csv")
	df_utils.HandlerError(err)

	defer cf.Close()
	w := csv.NewWriter(cf)

	for idx,record := range data {
		// Save header
		if idx == 0 {
			err = w.Write(record)
			df_utils.HandlerError(err)
			continue
		}
		
		outRecord := make([]string, 2)
		score, err := strconv.ParseFloat(strings.Split(record[0], "-")[0], 64)
		df_utils.HandlerError(err)
		outRecord[0] = strconv.FormatFloat((score - scoreMin)/(scoreMax - scoreMin), 'f', 4, 64)

		rate , err := strconv.ParseFloat(strings.TrimSuffix(record[1], "%") , 64)
		df_utils.HandlerError(err)
		if rate <= 12.0 {
			outRecord[1] = "1.0"
			df_utils.HandlerError(w.Write(outRecord))
			continue
		}
		outRecord[1] = "0.0"
		df_utils.HandlerError(w.Write(outRecord))
	}

	w.Flush()
	df_utils.HandlerError(w.Error())
}
