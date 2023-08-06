package main

import (
	"fmt"
	"math"
	"os"

	//"github.com/kniren/gota/dataframe"
	"github.com/go-gota/gota/dataframe"
	"github.com/montanaflynn/stats"
	"gonum.org/v1/gonum/floats"
	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/stat"
)

func basic() {

	a := mat.NewDense(3,3, []float64{1, 2, 3, 0, 4, 5, 0, 0, 6})
	fa := mat.Formatted(a, mat.Prefix(" "))
	fmt.Printf("mat = %v\n\n", fa)

	fmt.Printf("(0-1): %f\n", a.At(0,2))

	fmt.Println(mat.Col(nil, 0, a))
	fmt.Println(mat.Row(nil, 1, a))
}

func operations() {
   // Create two matrices of the same size, a and b.
   a := mat.NewDense(3, 3, []float64{1, 2, 3, 0, 4, 5, 0, 0, 6})
   b := mat.NewDense(3, 3, []float64{8, 9, 10, 1, 4, 2, 9, 0, 2})
   // Create a third matrix of a different size.
   c := mat.NewDense(3, 2, []float64{3, 2, 1, 4, 0, 8})

   d := mat.NewDense(3,3, nil)
   d.Add(a, b)
   fmt.Printf("d = %v\n", mat.Formatted(d, mat.Prefix(" ")))


   var f mat.Dense

   f.Mul(a,c)
   fmt.Printf("f = %v\n",mat.Formatted(&f, mat.Prefix("    "), mat.Squeeze()))

    // Raising a matrix to a power.
   row,col := a.Dims()
   g := mat.NewDense(row, col, nil)
   g.Pow(a, 5)
   fg := mat.Formatted(g, mat.Prefix("          "))
   fmt.Printf("g = a^5 = %0.4v\n\n", fg)
}

func applyFunctionToEachElement() {
   a := mat.NewDense(3, 3, []float64{4, 9, 16, 25, 36, 49, 0, 0, 64})
   raizCuadrada := func(_,_ int,v float64) float64 {
	   return math.Sqrt(v)
   }

   a.Apply(raizCuadrada, a)
   fmt.Printf(" Apply function a = %v\n",mat.Formatted(a, mat.Prefix("    "), mat.Squeeze()))
}

func transposeDeterminantInverse() {
   a := mat.NewDense(3, 3, []float64{4, 9, 16, 25, 36, 49, 0, 0, 64})

   // original matrix
   fmt.Printf("\n original a= %v\n", mat.Formatted(a, mat.Prefix(" ")))
   // transpose of the matrix
   fmt.Printf("\ntranspose a= %v\n", mat.Formatted(a.T(), mat.Prefix(" ")))

   // determinant of the matrix
   fmt.Printf("\n determinant a= %f\n", mat.Det(a))

   // inverse of the matrix
   err := a.Inverse(a)
   if err != nil {
	   panic(err)
   }

   fmt.Printf("\n inverse a= %v\n", mat.Formatted(a, mat.Prefix(" ")))
}

func meanModeMedian() {
	irisFile, err := os.Open("./iris.csv")
	if err != nil {
		panic(err)
	}
	defer irisFile.Close()

	// create dataframe from csv
	irisDF := dataframe.ReadCSV(irisFile)
	fmt.Println(irisDF.Names())
	fmt.Println(irisDF.Ncol())
	fmt.Println(irisDF.Nrow())
	fmt.Println(irisDF.Types())
	r,c := irisDF.Dims()
	fmt.Printf("row= %d, col= %d\n",r,c)

	// Get the float values from the "sepal_length" column as
   	// we will be looking at the measures for this variable.
	sepalLength := irisDF.Col("sepal.length").Float()
	meanVal := stat.Mean(sepalLength, nil)
	modeVal, modeCount := stat.Mode(sepalLength, nil)
	medianVal, err := stats.Median(sepalLength)
	if err != nil {
		panic(err)
	}
	fmt.Printf("mean: %f\n", meanVal)
	fmt.Printf("mode val: %f, mode count: %f\n", modeVal, modeCount)
	fmt.Printf("media: %f\n", medianVal)
}

func maxMinRangeVarianceStandardDesviationQuantiles() {
	irisFile, err := os.Open("./iris.csv")
	if err != nil {
		panic(err)
	}
	defer irisFile.Close()

	irisDF := dataframe.ReadCSV(irisFile)
	// Get the float values from the "sepal_length" column as
   	// we will be looking at the measures for this variable.
   	sepalLength := irisDF.Col("petal.length").Float()

	min := floats.Min(sepalLength)
	max := floats.Max(sepalLength)
	rangeValue := max - min
	varianza := stat.Variance(sepalLength, nil)
	standardDev := stat.StdDev(sepalLength, nil)
	
	inds := make([]int, len(sepalLength))
	floats.Argsort(sepalLength, inds)
	quant25 := stat.Quantile(0.25, stat.Empirical, sepalLength, nil)
	quant50 := stat.Quantile(0.50, stat.Empirical, sepalLength, nil)
	quant75 := stat.Quantile(0.75, stat.Empirical, sepalLength, nil)

	fmt.Println("*********************************************************************************************")
	fmt.Println("Summary Statistics: ")
	fmt.Printf("Max value: %f\n", max)
	fmt.Printf("Min value: %f\n", min)
	fmt.Printf("Range value: %f\n", rangeValue)
	fmt.Printf("Varianza: %f\n", varianza)
	fmt.Printf("Stadard desviation: %f\n", standardDev)
	fmt.Printf("Quantil 25: %f\n", quant25)
	fmt.Printf("Quantil 50: %f\n", quant50)
	fmt.Printf("Quantil 75: %f\n", quant75)
	fmt.Println("*********************************************************************************************")
}

func main() {
	basic()
	operations()
	applyFunctionToEachElement()
	transposeDeterminantInverse()
	meanModeMedian()
	maxMinRangeVarianceStandardDesviationQuantiles()
}
