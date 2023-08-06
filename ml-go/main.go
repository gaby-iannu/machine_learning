package main

import (
	"fmt"

	"gonum.org/v1/gonum/blas/blas64"
	"gonum.org/v1/gonum/floats"
	"gonum.org/v1/gonum/mat"
)

func main() {
   // Initialize a couple of "vectors" represented as slices.
   vectorA := []float64{11.0, 5.2, -1.3}
   vectorB := []float64{-7.2, 4.2, 5.1}

   // Compute the dot product of A and B
   // (https://en.wikipedia.org/wiki/Dot_product).
   dotProduct := floats.Dot(vectorA, vectorB)
   fmt.Printf("The dot product of A and B is: %0.2f\n", dotProduct)

   // Scale each element of A by 2
   floats.Scale(2, vectorA)
   fmt.Printf("Scaling A by 2 gives: %v\n", vectorA)

   // Compute the norm/length of B.
   normB := floats.Norm(vectorB, 3)
   fmt.Printf("The norm/length of B is: %0.2f\n", normB)


    // Initialize a couple of "vectors" represented as slices.
    vecA := mat.NewVecDense(3, []float64{11.0, 5.2, -1.3})
    vecB := mat.NewVecDense(3, []float64{-7.2, 4.2, 5.1})
   // Compute the dot product of A and B
   // (https://en.wikipedia.org/wiki/Dot_product).
   dotProduct = mat.Dot(vecA, vecB)
   fmt.Printf("The dot product of A and B is: %0.2f\n", dotProduct)
   // Scale each element of A by 1.5.
   vecA.ScaleVec(1.5, vecA)
   fmt.Printf("Scaling A by 1.5 gives: %v\n", vectorA)

   // Compute the norm/length of B.
   normB = blas64.Nrm2(vecB.RawVector())
   fmt.Printf("The norm/length of B is: %0.2f\n", normB)
}
