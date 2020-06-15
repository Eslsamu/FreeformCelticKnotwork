package main

import (
	"C"
	"bufio"
	"encoding/json"
	"fmt"
	"gonum.org/v1/gonum/diff/fd"
	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/optimize"
	"image"
	"image/color"
	"image/png"
	_ "io/ioutil"
	"log"
	"math"
	"math/rand"
	"os"
	"regexp"
	"strconv"
	"sync"
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
		return 153/256 - 9/8 * u + math.Pow(19/24*u, 3) - math.Pow(5/16*u, 4)
	} else {
		return 0
	}
}

//
func imageFeatureForce(atom []float64, pixel float64, x int, y int) float64{
	dist := distance(atom, []float64{float64(x),float64(y)})
	u := dist / d //normalized distance
	if 0 < u && u < 3/2 {
		return -1 * (pixel) * (1 / (1 + u)) //-1 - 0
	} else {
		return 0
	}
}

//calculate total potential energy of lattice
func totalPotentialEnergy(lattice *mat.Dense, img *image.Gray) float64 {
	r, c := lattice.Dims()
	latticeSize := float64(r * c)
	P := 0.0

	var wg sync.WaitGroup

	res := make(chan float64,21)
	//t := time.Now()
	for i := 0; i < r; i++ {
		wg.Add(1)
		go func(i int, c chan <- float64) {
			defer wg.Done()
			atom := mat.Row(nil, i, lattice)
			atomicPotentialField := 0.0
			for j := 0; j < r; j++ {
				if i != j {
					atomicPotentialField += (1 - beta) * atomicPotential(atom, mat.Row(nil, j, lattice))
				}
			}

			imagePotentialField := 0.0
			//forces only apply to iamge features in area around atom dependent on nominal distance d
			area := image.Rect(int(atom[0]-d),int(atom[1]-d),int(atom[0]+d),int(atom[1]+d))
			areaSize := float64(area.Size().X*area.Size().Y)
			for y := area.Min.Y; y < area.Max.Y; y++ {
				for x := area.Min.Y; x < area.Max.X; x++ {
					pixel := img.GrayAt(x, y)
					val,_,_,_ := pixel.RGBA()
					imagePotentialField += imageFeatureForce(atom, float64(val)/float64(^uint16(0)), x, y)
				}
			}
			appendToFile("test",fmt.Sprint(i,atomicPotentialField,imagePotentialField)+"\n")
			//normalize the potentials by dividing through the amount of atoms/pixels
			res <- 0.5 * (atomicPotentialField/latticeSize + beta * imagePotentialField/areaSize)
		}(i, res)

	}
	wg.Wait()

	for i:=0;i<r;i++{
		P += <-res
	}
	//fmt.Println(time.Since(t))
	return P
}

func pythonList2Array(list string) []float64 {
	re := regexp.MustCompile("[0-9.]+")
	numbers := re.FindAllString(list, -1)
	var data []float64
	for _, num := range numbers {
		value, err := strconv.ParseFloat(num, 64)
		if err != nil {
			panic(err)
		}
		data = append(data, value)
	}
	return data
}

func readImage(filePath string) (image.Image, error) {
	f, err := os.Open(filePath)
	if err != nil {
		return nil, err
	}
	image,  err := png.Decode(f)
	return image, err
}

func appendToFile(filename string,s string) {
	// If the file doesn't exist, create it, or append to the file
	f, err := os.OpenFile(filename, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
	if err != nil {
		log.Fatal(err)
	}
	if _, err := f.Write([]byte(s)); err != nil {
		log.Fatal(err)
	}
	if err := f.Close(); err != nil {
		log.Fatal(err)
	}
}

type Params struct {
	Atoms           string
	Iterations      int
	NominalDistance float64
	Beta 			float64
}


//TODO works only for png right now
func main() {
	//read stdin for parameters
	reader := bufio.NewReader(os.Stdin)
	text, _ := reader.ReadString('\n')

	//json input to struct
	var params Params
	err := json.Unmarshal([]byte(text), &params)
	if err != nil {
		fmt.Println(err)
	}

	appendToFile("params", fmt.Sprint(params))

	//parameters
	data := pythonList2Array(params.Atoms)

	appendToFile("params", fmt.Sprint(data))


	//read image and convert it to gray 0-255
	img, err := readImage("eyeball.png")
	gray := image.NewGray(img.Bounds())
	b := img.Bounds()
	for y := 0; y < b.Max.Y; y++ {
		for x := 0; x < b.Max.X; x++ {
			oldPixel := img.At(x, y)
			pixel := color.GrayModel.Convert(oldPixel)
			gray.Set(x,y,pixel)
		}
	}

	fcn := func(data []float64) float64 {
		lattice := mat.NewDense(len(data)/2, 2, data)
		return totalPotentialEnergy(lattice,gray)
	}

	grad := func(grad,data []float64) {
		fd.Gradient(grad, fcn, data, nil)
	}

	problem := optimize.Problem{
		Func: fcn,
		Grad: grad,
	}

	settings := &optimize.Settings{
		FuncEvaluations: 10000,
		GradientThreshold: 0.001,
	}

	method := &optimize.LBFGS{}

	result, err := optimize.Minimize(problem, data, settings, method)

	if err != nil {
		log.Fatal("err ",err)
	}
	if err = result.Status.Err(); err != nil {
		log.Fatal("err ",err)
	}

	bytes, err := json.Marshal(result)
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println(string(bytes))


}

