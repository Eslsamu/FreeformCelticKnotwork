package main

import (
	"C"
	"fmt"
	"gonum.org/v1/gonum/mat"
	"image/png"
	"math"
	"math/rand"
	"os/exec"
	"regexp"
	"strconv"
	"time"
)

var (
	d    = 10.0 //nominal distance
	beta = 0.5  // image-regularity tradeoff
)

func distance(v1 []float64, v2 []float64) float64 {
	return math.Sqrt(math.Pow(v1[0]-v2[0], 2) + math.Pow(v1[1]-v2[1], 2))
}

//adds random noise to the lattice with standard deviation "std"
func randomPerturb(lattice *mat.Dense, std float64) *mat.Dense {
	r, c := lattice.Dims()
	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			val := lattice.At(i, j) + rand.NormFloat64()*std
			lattice.Set(i, j, val)
		}
	}
	return lattice
}

func atomicPotential(atom1 []float64, atom2 []float64) float64 {
	dist := distance(atom1, atom2)
	u := dist / d //normalized distance

	if 0 < u && u < 3/2 {
		return 153/256 - 9/8*u + math.Pow(19/24*u, 3) - math.Pow(5/16*u, 4)
	} else {
		return 0
	}
}

//calculate total potential energy of lattice
func totalPotentialEnergy(lattice *mat.Dense) float64 {
	r, _ := lattice.Dims()

	t := time.Now()
	P := 0.0

	res := make(chan float64)
	for i := 0; i < r; i++ {
		go func(i int, c chan<- float64) {
			atom := mat.Row(nil, i, lattice)
			atomicPotentialField := 0.0
			for j := 0; j < r; j++ {
				if i != j {
					atomicPotentialField += (1 - beta) * atomicPotential(atom, mat.Row(nil, j, lattice))
				}
			}
			res <- atomicPotentialField
		}(i, res)
	}

	P = <-res
	fmt.Println(time.Since(t))
	return P
}

//This function converts a lattice in form of a python list that was printed to the console
//to a mat.Dense by processing each number appearing in the string.
func pythonList2Dense(list string) *mat.Dense {
	re := regexp.MustCompile("[0-9]+")
	numbers := re.FindAllString(list, -1)
	var data []float64
	for _, num := range numbers {
		value, err := strconv.ParseFloat(num, 64)
		if err != nil {
			panic(err)
		}
		data = append(data, value)
	}
	return mat.NewDense(len(data)/2, 2, data)
}

//TODO works only for png right now
func main() {
	//initialize lattice
	cmd := exec.Command("python3", "-c", "import mesh_init; mesh_init.init_pixels_go('eyeball.png')")
	//fmt.Println(cmd.Args)
	out, err := cmd.CombinedOutput()
	if err != nil {
		fmt.Println(1, err)
	}
	lattice := pythonList2Dense(string(out))

	//perturb lattice
	lattice = randomPerturb(lattice, 0.1)

	//load image
	png.load()

	//calculate potential energy
	totalPotentialEnergy(lattice)
}
