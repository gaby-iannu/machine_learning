package df_utils

import (
	"bufio"
	"fmt"
	"image/color"
	"math"
	"math/rand"
	"os"
	"strconv"
	"time"

	"github.com/go-gota/gota/dataframe"
	"github.com/gonum/matrix/mat64"
	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/vg"
)


func HandlerError(err error) {
	if err != nil {
		panic(err)
	}
}

func CreateDataFrame(pathToCsv string) dataframe.DataFrame {
	var df dataframe.DataFrame

	f, err := os.Open(pathToCsv)
	HandlerError(err)
	defer f.Close()
	df = dataframe.ReadCSV(f)

	return df
}

func CreateTrainingAndTestSet(df dataframe.DataFrame, n1, n2 int) (dataframe.DataFrame, dataframe.DataFrame){
	var dfTraining, dfTest dataframe.DataFrame
	
	trainingIdx, testIdx := trainigAndTestIndex(df, n1, n2)
	
	dfTraining = df.Subset(trainingIdx)
	dfTest = df.Subset(testIdx)

	return dfTraining, dfTest
}

func trainigAndTestIndex(df dataframe.DataFrame, n1, n2 int) ([]int, []int) {
	trainingNum, testNum := trainingAndTestNum(df, n1, n2)
	trainingIdx := make([]int, trainingNum)
	testIdx := make([]int, testNum)

	// Enumerate the training indices
	for i:=0; i<trainingNum; i++ {
		trainingIdx[i] = i
	}
	// Enumerate the test indices
	for i:=0; i<testNum; i++ {
		testIdx[i] = trainingNum + i
	}

	return trainingIdx, testIdx
}

func trainingAndTestNum(df dataframe.DataFrame, n1, n2 int) (int, int) {
	var trainingNum, testNum int 

	trainingNum = (n1 * df.Nrow()) / n2
	testNum = df.Nrow() / n2

	if trainingNum + testNum < df.Nrow() {
		trainingNum++
	}

	return trainingNum, testNum
}

func SaveDataFrameToCSV(path string, df dataframe.DataFrame) {
	f, err := os.Create(path)
	defer f.Close()
	HandlerError(err)
	w := bufio.NewWriter(f)
	err = df.WriteCSV(w)
	HandlerError(err)
}

func ParseFloat(sVal string) float64 {
	val, err := strconv.ParseFloat(sVal, 64)
	HandlerError(err)
	return val 
}
func OpenFile(file string) *os.File {
	f, err	:= os.Open(file)
	HandlerError(err)
	return f
}

func RegressionLinearPredict(m, b, x float64) float64 {
	return (m*x + b)
}

func CreateHistogram(df dataframe.DataFrame) []string{
	var fileName []string

	fileName = make([]string, df.Ncol())
	for j, col := range df.Names() {
		plotVals := make(plotter.Values, df.Nrow())
		for i, floatVal := range df.Col(col).Float() {
			plotVals[i] = floatVal
		}
		p := plot.New()

		p.Title.Text = fmt.Sprintf("Histogram of a %s", col)
		h,err := plotter.NewHist(plotVals, 16)
		HandlerError(err)

		h.Normalize(1)
		p.Add(h)
		fileName[j] = fmt.Sprintf("%s%s",col, "_hist.png")
		err = p.Save(4*vg.Inch, 4*vg.Inch, fileName[j])
		HandlerError(err)
	}

	return fileName
}

func LogisticRegression(features *mat64.Dense, labels []float64, numSteps int, learningRate float64) []float64 {
	var weights []float64

	_,numWeights := features.Dims()
	weights = make([]float64, numWeights)
	s := rand.NewSource(time.Now().UnixNano())
	r := rand.New(s)

	for i,_ := range weights {
		weights[i] = r.Float64()
	}
	
	for i:=0; i<numSteps; i++ {
		var sumError float64
		for j,label := range labels {
			
			featureRow := mat64.Row(nil, j, features)
			pred := logistic(featureRow[0]*weights[0])
			// featureRow[1]*weights[1]
			predError := label - pred
			sumError += math.Pow(predError, 2)
			for k:=0; k<len(featureRow); k++ {
				weights[k] += learningRate*predError*pred*(1-pred)*featureRow[k]  
			}
		}
	}
	return weights
}

func logistic(x float64) float64 {
	return 1 / (1 + math.Exp(-x))
}

func LogisticRegresionPredict(score float64) float64 {

	m1 := -0.48
	m2 := 1.54
	p := 1 / (1 + math.Exp(-m1 * score + m2))

	if p > 0.5 {
		return 1.0
	}

	return 0.0
}

/*
	pathToCsv: Nombre del CSV
	yColumnName: Nombre de la columna Y a filtrar del df 
	xColumnName: Nombre de la columna X a filtrar del df 
	xLabel: Texto eje X
	yLabel: Texto eje Y
	plotterName: Nombre de la imagen guardada
*/
func CreateScatterPlot(pathToCsv, yColumnName, xColumnName, xLabel, yLabel, plotterName  string) {
	df := CreateDataFrame(pathToCsv)
	yVals := df.Col(yColumnName).Float()

	pts := make(plotter.XYs, df.Nrow())

	for i,floatVal := range df.Col(xColumnName).Float() {
		pts[i].X = floatVal
		pts[i].Y = yVals[i]
	}

	p := plot.New()
	p.X.Label.Text = xLabel
	p.Y.Label.Text = yLabel
	p.Add(plotter.NewGrid())

	scatter, err := plotter.NewScatter(pts)
	HandlerError(err)

	scatter.GlyphStyle.Color = color.RGBA{R:255, B:128, A:255}
	scatter.GlyphStyle.Radius = vg.Points(3)

	p.Add(scatter)
	err = p.Save(4*vg.Inch, 4*vg.Inch, plotterName)
	HandlerError(err)
}

// func a(b func(v1, v2 []float64, row int)plotter.XYs) {
// 	var pts plotter.XYs

// 	if b == nil {
// 		b = func(v1, v2 []float64, row int)plotter.XYs {
// 			p := make(plotter.XYs, row)
			
// 			for i,v := range v1 {
// 				p[i].X = v   
// 				p[i].Y = v2[i]
// 			}

// 			return p
// 		}
// 	}

// 	pts = b(nil, nil, 1)
// 	fmt.Println(pts)
// }


