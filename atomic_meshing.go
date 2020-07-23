package main

import (
	"C"
	"bufio"
	"encoding/json"
	"fmt"
	"gonum.org/v1/gonum/diff/fd"
	"gonum.org/v1/gonum/optimize"

	//"gonum.org/v1/gonum/diff/fd"
	//"gonum.org/v1/gonum/mat"
	//"gonum.org/v1/gonum/optimize"
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
	d    = 		10.0 //nominal distance
	beta = 		0.5  //image-regularity tradeoff
	gray *image.Gray //image
	atoms []float64 //atoms to mesh
)

func distance(v1 []float64, v2 []float64) float64 {
	dist := math.Sqrt(math.Pow(v1[0]-v2[0], 2) + math.Pow(v1[1]-v2[1], 2))
	return dist
}

//adds random noise to the lattice with standard deviation "std"
func randomPerturb(atoms []float64, std float64) []float64 {
	for i := 0; i < len(atoms); i++ {
		atoms[i] += rand.NormFloat64() * std
	}
	return atoms
}

func atomicPotential(atom1 []float64, atom2 []float64) float64 {
	dist := distance(atom1, atom2)
	u := dist / d //normalized distance
	/*appendToFile("log", "distance between" + fmt.Sprint(atom1) + " " +
		fmt.Sprint(atom2) + fmt.Sprint(dist) +"\n" +
		" normalized distance: " + fmt.Sprint(u) +"\n")*/
	if 0 < u && u < 1.5 {
		return 153.0/256.0 - 9.0/8.0 * u + 19.0/24.0 * math.Pow(u, 3) - 5.0/16.0 * math.Pow(u, 4)
	} else {
		return 0
	}
}

//TODO we want things to flow into the black features so invert the image, 1 has to be gray 0 white
func imageFeatureForce(atom []float64, pixel float64, x int, y int) float64{
	dist := distance(atom, []float64{float64(x),float64(y)})
	u := dist / d //normalized distance
	if 0 < u && u < 3.0/2.0 {
		return -1 * (pixel) * (1. / (1. + u)) //-1 - 0
	} else {
		return 0
	}
}

//calculate total potential energy of lattice
func totalPotentialEnergy(x []float64) float64{
	r := len(x)/2
	P := 0.0

	var wg sync.WaitGroup

	res := make(chan float64,r+1)
	//t := time.Now()
	for i := 0; i < r; i++ {
		wg.Add(1)
		go func(i int, c chan <- float64) {
			defer wg.Done()
			atom := x[i*2:i*2+2]

			atomicPotentialField := 0.0
			for j := 0; j < r; j++ {
				if i != j {
					atom2 := x[j*2:j*2+2]
					atomicPotentialField += (1 - beta) * atomicPotential(atom, atom2)
				}
			}

			imagePotentialField := 0.0
			//forces only apply to image features in area around atom dependent on nominal distance d
			area := image.Rect(int(atom[0]-5),int(atom[1]-5),int(atom[0]+5),int(atom[1]+5))
			areaSize := float64(area.Size().X*area.Size().Y)
			for y := area.Min.Y; y < area.Max.Y; y++ {
				for x := area.Min.X; x < area.Max.X; x++ {
					pixel := gray.GrayAt(x, y)
					val,_,_,_ := pixel.RGBA()
					imagePotentialField += imageFeatureForce(atom, float64(val)/float64(^uint16(0)), x, y)
				}
			}
			appendToFile("log","Atom | apf | ipf "+fmt.Sprint(i,x[i*2:i*2+1] ,atomicPotentialField/float64(r),imagePotentialField/areaSize)+"\n")
			//normalize the potentials by dividing through the amount of atoms/pixels
			res <- 0.5 * (atomicPotentialField/float64(r) + beta * imagePotentialField/areaSize)
		}(i, res)
	}
	wg.Wait()
	appendToFile("log","done \n")
	for i:=0;i<r;i++{
		P += <-res
	}
	appendToFile("log",fmt.Sprint(P) + " <- total potential field \n")
	//fmt.Println(time.Since(t))
	return P
}

func pythonList2Vector(list string) []float64 {
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
	Image 			string
}

type Recorder struct {

}

func (r Recorder) Init() error{
	return nil
}

func (r Recorder) Record(loc *optimize.Location,op optimize.Operation,stats *optimize.Stats) error{
	appendToFile("log",fmt.Sprint("loc:",loc,"\n op:", op,"\n stats:", stats,"\n"))
	return nil
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
		appendToFile("log ", "Error: "+fmt.Sprint(err)+"\n")
		log.Fatal("err ",err)
	}

	appendToFile("log","----------- Atomic meshing ---------\n " +
		"Input parameters: " + fmt.Sprint(params) + "\n")

	//parameters
	atoms = pythonList2Vector(params.Atoms)
	d = params.NominalDistance
	beta = params.Beta
	imgFile := params.Image

	//read image and convert it to gray 0-255
	img, _ := readImage(imgFile)
	gray = image.NewGray(img.Bounds())
	b := img.Bounds()
	for y := 0; y < b.Max.Y; y++ {
		for x := 0; x < b.Max.X; x++ {
			oldPixel := img.At(x, y)
			pixel := color.GrayModel.Convert(oldPixel)
			gray.Set(x,y,pixel)
		}
	}

	appendToFile("log","Lattice size " + fmt.Sprint(len(atoms)/2) +"\n")
	appendToFile("log","Image dimensions " + fmt.Sprint(b) +"\n")

	fcn := func(data []float64) float64 {
		return totalPotentialEnergy(data)
	}

	diffSettings := &fd.Settings{
		Step: 0.001, //pixel shift for first difference
	}


	grad := func(grad,data []float64) {
		fd.Gradient(grad, fcn, data, diffSettings)
		appendToFile("log","Gradient: " + fmt.Sprint(grad) + "\n")
	}

	problem := optimize.Problem{
		Func: fcn,
		Grad: grad,
	}



	settings := &optimize.Settings{
		FuncEvaluations: 100000,
		GradientThreshold: 0.00001,
		Recorder: Recorder{},
	}

	appendToFile("log","optimization settings :" +fmt.Sprint(settings)+"\n")

	method := &optimize.LBFGS{
		Linesearcher: &optimize.Bisection{
			CurvatureFactor: 0.9,
			MinStep: 1,
		},
	}


	for true {
		PInit := totalPotentialEnergy(atoms)
		appendToFile("log","init function value: "+fmt.Sprint(PInit)+"\n")
		result, err := optimize.Minimize(problem, atoms, settings, method)
		appendToFile("log",fmt.Sprint(result))

		if err != nil {
			appendToFile("log", "Error1: "+fmt.Sprint(err)+"\n")
		}

		if err = result.Status.Err(); err != nil {
			appendToFile("log", "Error2: "+fmt.Sprint(err)+"\n")
		}
		atoms = result.X

		d := math.Abs(PInit- totalPotentialEnergy(atoms)) / math.Abs(PInit * 0.001)
		appendToFile("log","function value: "+fmt.Sprint(totalPotentialEnergy(atoms))+"\n")
		appendToFile("log","Delta: "+fmt.Sprint(d)+"\n")
		appendToFile("log", "Atoms: "+fmt.Sprint(atoms))
		if d <= 1  {
			appendToFile("log","----------- finished meshing ----------"+"\n")
			bytes, err := json.Marshal(result)
			if err != nil {
			appendToFile("log", "Error3: "+fmt.Sprint(err)+"\n")
			}
			appendToFile("log", "Result "+string(bytes)+"\n")
			fmt.Println(string(bytes))
			break
		} else {
			randomPerturb(atoms,1)
			appendToFile("log", "perturbed atoms: "+fmt.Sprint(atoms))
		}
	}
}

