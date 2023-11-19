// Package main provides ...
package main

import (
	"encoding/csv"
	"errors"
	"fmt"
	"math"
	"math/rand"
	"os"
	"strconv"
	"time"

	"gonum.org/v1/gonum/mat"

	"github.com/gaby-iannu/df_utils"
	"github.com/gonum/floats"
)

// neuralNet contains all of the information
// that defines a trained neural network
type neuralNetwork struct {
	config neuralNetConfig
	wHidden *mat.Dense
	bHidden *mat.Dense
	wOut *mat.Dense
	bOut *mat.Dense
}

type neuralNetConfig struct {
	inputNeurons int
	outputNeurons int
	hiddenNeurons int 
	numEpochs int
	learningRate float64
}

func newNeuralNetwork(config neuralNetConfig) *neuralNetwork {
	return &neuralNetwork{
		config: config,
	}
}

// sigmoid implements the sigmoid function
// for use in activation functions.
func sigmoid(x float64) float64 {
       return 1.0 / (1.0 + math.Exp(-x))
}

func sigmoidPrime(x float64)float64 {
	return x * (1.0 - x)
}

// sumAlongAixs sums a matrix along a 
// particular dimension, persevering the
// other dimension.
func sumAlongAxis(axis int, m *mat.Dense) (*mat.Dense, error) {
	nRow,nCol := m.Dims()
	var output *mat.Dense

	switch axis{
	case 0:
		data := make([]float64, nCol)
		for i:=0; i<nCol; i++ {
			col := mat.Col(nil, i, m)
			data[i] = floats.Sum(col)
		}
		output = mat.NewDense(1, nCol, data)
		break
	case 1:
		data := make([]float64, nRow)
		for i:=0; i<nRow; i++ {
			row := mat.Row(nil, i, m)
			data[i] = floats.Sum(row)
		}
		output = mat.NewDense(nRow, 1, data)
		break
	default:
		return nil, errors.New("invalid axis, must be 0 or 1")
	}

	return output, nil
}

func (n *neuralNetwork) feedForwardProcess(wHidden, bHidden, wOut, bOut, output, x *mat.Dense) *mat.Dense {
	hiddenLayerInput := new(mat.Dense)
	hiddenLayerInput.Mul(x, wHidden)
	addBHidden := func(_, col int, v float64) float64{
		return v + bHidden.At(0, col)
	}
	hiddenLayerInput.Apply(addBHidden, hiddenLayerInput)
	hiddenLayerActivations := new(mat.Dense)
	applySigmoid := func(_, _ int, v float64) float64 { 
		return sigmoid(v) 
	}
	hiddenLayerActivations.Apply(applySigmoid, hiddenLayerInput)
	outputLayerInput := new(mat.Dense)
	outputLayerInput.Mul(hiddenLayerActivations, wOut)

	addBOut := func(_, col int, v float64) float64 { 
		return v + bOut.At(0, col)
	}
	outputLayerInput.Apply(addBOut, outputLayerInput)

	output.Apply(applySigmoid, outputLayerInput)

	return hiddenLayerActivations
}

func (n *neuralNetwork) backpropagation(hiddenLayerActivations, wOut, output, y *mat.Dense) (*mat.Dense, *mat.Dense) {
	networkError := new(mat.Dense)
	networkError.Sub(y, output)
	slopeOutputLayer := new(mat.Dense)
	applySigmoidPrime := func(_, _ int, v float64) float64 { 
		return sigmoidPrime(v) 
	}
	slopeOutputLayer.Apply(applySigmoidPrime, output)
	slopeHiddenLayer := new(mat.Dense)
	slopeHiddenLayer.Apply(applySigmoidPrime, hiddenLayerActivations)

	dOutput := new(mat.Dense)

	dOutput.MulElem(networkError, slopeOutputLayer)
	errorAtHiddenLayer := new(mat.Dense)
	errorAtHiddenLayer.Mul(dOutput, wOut.T())

	dHiddenLayer := new(mat.Dense)
	dHiddenLayer.MulElem(errorAtHiddenLayer, slopeHiddenLayer)

	return dOutput, dHiddenLayer
}

func (n *neuralNetwork) adjustParameters(hiddenLayerActivations, wHidden, bHidden, dHiddenLayer, dOutput, bOut, x *mat.Dense) {
	wOutAdj := new(mat.Dense)
	wOutAdj.Mul(hiddenLayerActivations.T(), dOutput)
	wOutAdj.Scale(n.config.learningRate, wOutAdj)

	bOutAdj,err := sumAlongAxis(0, dOutput)
	df_utils.HandlerError(err)
	bOutAdj.Scale(n.config.learningRate, bOutAdj)
	bOut.Add(bOut, bOutAdj)
	wHiddenAdj := new(mat.Dense)
	wHiddenAdj.Mul(x.T(), dHiddenLayer)
	wHiddenAdj.Scale(n.config.learningRate, wHiddenAdj)
	wHidden.Add(wHidden, wHiddenAdj)
	bHiddenAdj, err := sumAlongAxis(0, dHiddenLayer)
	df_utils.HandlerError(err)
	bHiddenAdj.Scale(n.config.learningRate, bHiddenAdj)
	bHidden.Add(bHidden, bHiddenAdj)
}

func (n *neuralNetwork) initHiddenOutRaw() ([]float64, []float64, []float64, []float64){
	randSource := rand.NewSource(time.Now().UnixNano())
	randGen := rand.New(randSource)
	wHiddenRaw := make([]float64, n.config.hiddenNeurons * n.config.inputNeurons)
	bHiddenRaw := make([]float64, n.config.hiddenNeurons)
	wOutRaw := make([]float64, n.config.outputNeurons * n.config.hiddenNeurons)
	bOutRaw := make([]float64, n.config.hiddenNeurons)
	for _,param := range [][]float64{wHiddenRaw, bHiddenRaw, wOutRaw, bOutRaw} {
		for i := range param {
			param[i] = randGen.Float64()
		}
	}

	return wHiddenRaw, bHiddenRaw, wOutRaw, bOutRaw
}

func (n *neuralNetwork) initHiddenOut(wHiddenRaw, bHiddenRaw, wOutRaw, bOutRaw []float64) (*mat.Dense, *mat.Dense, *mat.Dense, *mat.Dense) {
	wHidden := mat.NewDense(n.config.inputNeurons, n.config.hiddenNeurons, wHiddenRaw)
	bHidden := mat.NewDense(1, n.config.hiddenNeurons, bHiddenRaw)
	wOut := mat.NewDense(n.config.hiddenNeurons, n.config.outputNeurons, wOutRaw)
	bOut := mat.NewDense(1, n.config.outputNeurons, bOutRaw)
	return wHidden, bHidden, wOut, bOut
}

func (n *neuralNetwork) update(wHidden, bHidden, wOut, bOut *mat.Dense) {
	n.wHidden = wHidden
	n.bHidden = bHidden
	n.wOut = wOut
	n.bOut = bOut
}


func (n *neuralNetwork) train(x,y *mat.Dense) error {
	wHiddenRaw,bHiddenRaw,wOutRaw,bOutRaw := n.initHiddenOutRaw()
	wHidden, bHidden, wOut, bOut := n.initHiddenOut(wHiddenRaw, bHiddenRaw, wOutRaw, bOutRaw)

	// Define the output of the neural network
	output := new(mat.Dense)

	for i:=0; i<n.config.numEpochs; i++ {
		// Complete the feed for forward proecess
		hiddenLayerActivations := n.feedForwardProcess(wHidden, bHidden, wOut, bOut, output, x)
		// Complete the backpropagation
		dOutput, dHiddenLayer := n.backpropagation(hiddenLayerActivations, wOut, output, y) 
		// Adjust the parameters
		n.adjustParameters(hiddenLayerActivations, wHidden, bHidden, dHiddenLayer, dOutput, bOut, x)
	}

	// Define our trained neural network
	n.update(wHidden, bHidden, wOut, bOut)
	return nil
}

func (n *neuralNetwork) print() {
	f := mat.Formatted(n.wHidden, mat.Prefix(" "))
	fmt.Printf("\nwHidden: %v\n", f)

	f = mat.Formatted(n.bHidden, mat.Prefix(" "))
	fmt.Printf("\nbHidden: %v\n", f)

	f = mat.Formatted(n.wOut, mat.Prefix(" "))
	fmt.Printf("\nwOut: %v\n", f)

	f = mat.Formatted(n.wOut, mat.Prefix(" "))
	fmt.Printf("\nbOut: %v\n\n", f)
}

func (n *neuralNetwork) predict(x *mat.Dense) (*mat.Dense, error) {
	if n.wHidden == nil || n.bHidden == nil || n.wOut == nil || n.bOut == nil {
		return nil, errors.New("the supplied neural network weights and biases are empty")
	}

	// Define the output of the neural network 
	output := new(mat.Dense)
	hiddenLayerInput := new(mat.Dense)
	hiddenLayerActivations := new(mat.Dense)
	outputLayerInput := new(mat.Dense)

	addBHidden := func(_, col int, v float64) float64 {
		return v + n.bHidden.At(0, col)
	}

	applySigmoid := func(_,_ int, v float64) float64 {
		return sigmoid(v)
	}

	addBOut := func(_, col int, v float64) float64 {
		return v + n.bOut.At(0, col)
	}

	hiddenLayerInput.Mul(x, n.wHidden)
	hiddenLayerInput.Apply(addBHidden, hiddenLayerInput)
	hiddenLayerActivations.Apply(applySigmoid, hiddenLayerInput)
	outputLayerInput.Mul(hiddenLayerActivations, n.wOut)
	outputLayerInput.Apply(addBOut, outputLayerInput)
	output.Apply(applySigmoid, outputLayerInput)
	return output, nil
}

func buildConfigCase1() neuralNetConfig {
	// Define our network architecture and
	// learning parameters
	return  neuralNetConfig{
		inputNeurons: 4,
		outputNeurons: 3,
		hiddenNeurons: 3,
		numEpochs: 500,
		learningRate: 0.3,
	}
}

func buildInputAndLabelsCase1() (*mat.Dense, *mat.Dense) {
	// Define our input attribute 
	input := mat.NewDense(3,4,[]float64{
		1.0, 0.0, 1.0, 0.0,
		1.0, 0.0, 1.0, 1.0,
		0.0, 1.0, 0.0, 1.0,
	})

	// Define our labels
	labels := mat.NewDense(3, 3, []float64{
				1.0, 0.0, 1.0,
				0.0, 0.0, 0.0,
				0.0, 0.0, 0.0,
			})
	return input, labels
}

func buildInputAndLabelsCase2(path string) (*mat.Dense,*mat.Dense) {
	f := df_utils.OpenFile(path)
	defer f.Close()

	reader := csv.NewReader(f)
	records,err := reader.ReadAll()
	df_utils.HandlerError(err)

	// inputsData and labelsData will hold all the
	// float values that will eventually be
	// used to form our matrices
	inputsData := make([]float64, 4*len(records))
	labelsData := make([]float64, 3*len(records))
	id := 0
	ld := 0
	for i, record := range records {
		if i == 0 {
			continue
		}
		
		// Loop over the float columns
		for j, v := range record {
			if j == len(record) - 1 {
				continue
			}
			value := df_utils.ParseFloat(v)
			if j == 4 || j == 5 || j == 6 {
				labelsData[ld] = value
				ld++
				continue
			}

			inputsData[id] = value
			id++
		}
	}

	// Form the matrices
	inputs := mat.NewDense(len(records),4, inputsData)
	labels := mat.NewDense(len(records),3, labelsData)
	return inputs, labels
}

func buildConfigCase2() neuralNetConfig {
	return neuralNetConfig {
		inputNeurons: 4,
		outputNeurons: 3,
		hiddenNeurons: 3,
		numEpochs: 5000,
		learningRate: 0.3,
	}
}

func buildTestInput() *mat.Dense {
	f := df_utils.OpenFile("./test.csv")
	defer f.Close()

	reader := csv.NewReader(f)
	records,err := reader.ReadAll()
	df_utils.HandlerError(err)

	inputsData := make([]float64, 4*len(records))

	for _,record := range records {

		// Loop over the float columns
		for j, v := range record {
			if j == len(record) - 1 {
				continue
			}
			value := df_utils.ParseFloat(v)

			inputsData[j] = value
		}
	}

	inputs := mat.NewDense(len(records),4, inputsData)
	return inputs
}

func accuracy(predictions, labels *mat.Dense) {

	// calculate accuracy of our model
	var truePosNeg int
	numPreds,_ := predictions.Dims()

	for i:=0; i<numPreds ; i++ {
		// Get the label
		labelRow := mat.Row(nil, i, labels)
		var species int 
		for j, label := range labelRow {
			if label == 1.0 {
				species = j
				break
			}
		}
		// Accumulate the true positive/negative count
		if predictions.At(i, species) == floats.Max(mat.Row(nil, i, predictions)) {
			truePosNeg++
		}
	}

	accuracyVal := float64(truePosNeg)/float64(numPreds)
	fmt.Printf("Accuracy: %.2f\n", accuracyVal)
}

func main() {
	var network *neuralNetwork

	if len(os.Args) != 2 {
		panic("there is not argument to execute (1|2)")
	}

	option,err := strconv.Atoi(os.Args[1])
	// fmt.Println(option)
	df_utils.HandlerError(err)

	switch option {
	case 1:
		// Train the neural network
		network = newNeuralNetwork(buildConfigCase1())
		err = network.train(buildInputAndLabelsCase1())
		df_utils.HandlerError(err)
		break
	case 2:
		network = newNeuralNetwork(buildConfigCase2())
		err = network.train(buildInputAndLabelsCase2("./train.csv"))
		df_utils.HandlerError(err)
		testInputs, testLabels := buildInputAndLabelsCase2("./test.csv")
		result,err := network.predict(testInputs)
		df_utils.HandlerError(err)
		accuracy(result, testLabels)
		break
	default:
		panic("incorrect argument")
	}

	// Output the weights that define our network
	network.print()
}
