// Package main provides ...
package main

import (
	"encoding/csv"
	"fmt"
	"io"

	"github.com/gaby-iannu/df_utils"
)

func main() {
	f := df_utils.OpenFile("../training_test/test.csv")
	defer f.Close()
	reader := csv.NewReader(f)
	
	var observed []float64
	var predicted []float64

	line := 1
	for {
		record, err := reader.Read()

		if err == io.EOF {
			break
		}

		if line == 1 {
			line++
			continue
		}

		predictedVal := df_utils.LogisticRegresionPredict(df_utils.ParseFloat(record[0]))
		observedVal := df_utils.ParseFloat(record[1])

		predicted = append(predicted, predictedVal)
		observed = append(observed, observedVal)
	}

	var truePosNeg int

	for i, obser := range observed {
		if obser == predicted[i] {
			truePosNeg++
		}
	}

	accuracy := float64(truePosNeg) / float64(len(observed))
	fmt.Printf("Accuracy: %0.2f\n", accuracy)
}
