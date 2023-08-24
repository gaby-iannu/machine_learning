// Package main provides ...
package main

import (
	"github.com/gaby-iannu/df_utils"
	"github.com/go-gota/gota/dataframe"
	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/vg"
)

func main() {
	f := df_utils.OpenFile("../create_training_test/Advertising.csv")
	defer f.Close()

	adDF := dataframe.ReadCSV(f)
	yVals := adDF.Col("Sales").Float()
	
	pts := make(plotter.XYs, adDF.Nrow())
	ptsPred := make(plotter.XYs, adDF.Nrow())

	for i, floatVal := range adDF.Col("TV").Float() {
		pts[i].X = floatVal
		pts[i].Y = yVals[i]
		ptsPred[i].X = floatVal
		ptsPred[i].Y = df_utils.RegressionLinearPredict(-0.0336, 34.8333, floatVal)
	}

	p := plot.New()
	p.X.Label.Text = "TV"
	p.Y.Label.Text = "Sales"
	p.Add(plotter.NewGrid())

	s,err := plotter.NewScatter(pts)
	df_utils.HandlerError(err)
	s.GlyphStyle.Radius = vg.Points(3)

	l, err := plotter.NewLine(ptsPred)
	df_utils.HandlerError(err)
	l.LineStyle.Width = vg.Points(1)
	l.LineStyle.Dashes = []vg.Length{vg.Points(5), vg.Points(5)}

	p.Add(s, l)
	df_utils.HandlerError(p.Save(4*vg.Inch, 4*vg.Inch, "regression_line.png"))
}
