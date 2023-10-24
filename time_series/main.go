package main

import (
	"bufio"
	"encoding/csv"
	"fmt"
	"image/color"
	"math"
	"os"
	"strconv"
	"strings"

	"github.com/gaby-iannu/df_utils"
	"github.com/sajari/regression"
	"gonum.org/v1/gonum/stat"
	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/plotutil"
	"gonum.org/v1/plot/vg"
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


func calculateAutocorreletion() {
	df := df_utils.CreateDataFrame(path_to_csv)
	passengers := df.Col("value").Float()
	fmt.Println("Autocorrelation:")
	for i:=1; i<11; i++ {
		// Shift the series
		adjusted := passengers[i:len(passengers)]
		lag := passengers[0:len(passengers)-i]

		// Calculate the autocorrelation
		ac := stat.Correlation(adjusted, lag, nil)
		fmt.Printf("Lag %d period: %0.2f\n", i, ac)
	}
}

func doBarChartWithAcf() {
	df := df_utils.CreateDataFrame(path_to_csv)
	passengers := df.Col("value").Float()
	p := plot.New()

	p.Title.Text = "Autocorrelation for AirPassengers"
	p.X.Label.Text = "Lag"
	p.Y.Label.Text = "ACF"
	p.Y.Min = 0
	p.Y.Max = 1
	w := vg.Points(3)

	numLags := 20
	pts := make(plotter.Values, numLags)

	for i:=1; i<=numLags; i++ {
		pts[i-1] = acf(passengers, i)
	}

	bars, err := plotter.NewBarChart(pts, w)
	df_utils.HandlerError(err)

	bars.LineStyle.Width = vg.Length(0)
	bars.Color = plotutil.Color(1)

	p.Add(bars)
	err = p.Save(8*vg.Inch, 4*vg.Inch, "acf.png")
	df_utils.HandlerError(err)
} 

func acf(x []float64, lag int) float64 {
	
	// shift the series
	xAdj := x[lag:len(x)]
	xLag := x[0:len(x)-lag]

	// Numerator will hold our accumulated numerator
	var numerator float64
	// Denominator will hold our accumulated denominator
	var denominator float64

	// Calculate the mean of our x values, which will be used 
	// in each termn of the autocorrelation
	xBar := stat.Mean(x, nil)

	// Calculate the numerator
	for i,xVal := range xAdj {
		numerator += ((xVal-xBar)*(xLag[i]-xBar))
	}

	// Calculate the denominator
	for _,xVal := range x {
		denominator += math.Pow(xVal-xBar, 2)
	}

	return numerator/denominator
}

func pacf(x []float64, lag int) float64 {
	var r regression.Regression
	
	r.SetObserved("x")
	for i:=0; i<lag; i++ {
		r.SetVar(i, "x"+strconv.Itoa(i))
	}

	xAdj := x[lag:len(x)]
	for i, xVal := range xAdj {
		laggedVariables := make([]float64, lag)
		for j:=1; j<=lag; j++ {
			laggedVariables[j-1] = x[lag+i-j]
		}
		r.Train(regression.DataPoint(xVal, laggedVariables))
	}
	r.Run()
	return r.Coeff(lag)
}

func calculatePartialAutocorrelation() {
	df := df_utils.CreateDataFrame(path_to_csv)
	passengers := df.Col("value").Float()
	fmt.Println("Calculate partial autocorrelation")
	for i:=0; i<=11; i++ {
		pac := pacf(passengers, i)
		fmt.Printf("Lag %d period: %0.2f\n", i, pac)
	}
}

func transformingToStationarySeries() {
	df := df_utils.CreateDataFrame(path_to_csv)
	passengersVals := df.Col("value").Float()
	timeVals := df.Col("time").Float()
	pts := make(plotter.XYs, df.Nrow()-1)

	var differenced [][]string
	differenced = append(differenced, []string{"time", "differenced_time"})

	for i:=1; i<len(passengersVals); i++ {
		pts[i-1].X = timeVals[i]
		pts[i-1].Y = passengersVals[i] - passengersVals[i-1]
		differenced = append(differenced, []string{strconv.FormatFloat(timeVals[i], 'f', -1, 64),
				strconv.FormatFloat(passengersVals[i]-passengersVals[i-1], 'f', -1, 64)})
	}

	p := plot.New()
	p.X.Label.Text = "time"
	p.Y.Label.Text = "differenced passengers"
	p.Add(plotter.NewGrid())
	l,err := plotter.NewLine(pts)
	df_utils.HandlerError(err)
	l.LineStyle.Width = vg.Points(1)
	l.LineStyle.Color = color.RGBA{B:255, A:255}
	p.Add(l)
	err = p.Save(10*vg.Inch, 4*vg.Inch, "diff_passengers.png")
	df_utils.HandlerError(err)

	f,err := os.Create("diff_series.csv")
	df_utils.HandlerError(err)
	defer f.Close()

	w := csv.NewWriter(f)
	w.WriteAll(differenced)
	df_utils.HandlerError(w.Error())
}

func menu() {
	
	var b strings.Builder
	b.WriteString("1 - Print dataframe AirPassengers.csv\n")
	b.WriteString("2 - Do plot \n")
	b.WriteString("3 - Calculate autocorrelation \n")
	b.WriteString("4 - Do bar chart with acf \n")
	b.WriteString("5 - Calculate partial autocorrelation \n")
	b.WriteString("6 - Transforming to a stationary series \n")
	b.WriteString("q - Quit \n")
	menu := b.String()

	scanner := bufio.NewScanner(os.Stdin)
	for {
		fmt.Println(menu)
		fmt.Print("Type here: ")
		scanner.Scan()
		txt := scanner.Text()
		if txt == "q" {
			os.Exit(0)
		}
		option,_ := strconv.Atoi(txt)
		switch option {
		case 1:
			printDF()
			break
		case 2:
			doPlot()
			break
		case 3:
			calculateAutocorreletion()
			break
		case 4:
			doBarChartWithAcf()
			break
		case 5:
			calculatePartialAutocorrelation()
			break
		case 6:
			transformingToStationarySeries()
			break
		}
	}
}

func main() {
	menu()
}
