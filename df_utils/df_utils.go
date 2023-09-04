package df_utils

import (
	"bufio"
	"fmt"
	"os"
	"strconv"

	"github.com/go-gota/gota/dataframe"
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