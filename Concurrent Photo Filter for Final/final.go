package main

//	Uncomment the Correct Image Format
import (
	"os"
	"log"
	"math"
	"time"
	"image"
	"image/color"
	// "image/png"
	// "image/gif"
	"image/jpeg"
)

///// Interfaces have the declaration of the function /////
type ImageSet interface {
	Set(x, y int, c color.Color) 
}

///// Converts original image using black and white filter /////
func BlackAndWhite(){

	file, err := os.Open("Original.jpg") // OS opens file
	img, err := jpeg.Decode(file) // Turns img to img object

	b := img.Bounds() // Bounds are to know the size of the image so new object can be created]

	imgSet := image.NewRGBA(b) // Instantiate new img object

	// Loop takes old RGB values, multiplies by luminosity values, averages them, and sets new RGB values
	for y := 0; y < b.Max.Y; y++ {
		for x := 0; x < b.Max.X; x++ {
			oldPixel := img.At(x, y)
			r, g, b, _ := oldPixel.RGBA() // Get old pixel's values
			lum := 0.299*float64(r) + 0.587*float64(g) + 0.114*float64(b) // Multiply by luminosity values
			pixel := color.Gray{uint8(lum / 256)} // Use the grey function for black and white filter, requires uint8
			imgSet.Set(x, y, pixel)
		}
    }

	outFile, err := os.Create("Black_and_White.jpg") // Save output img object to img file on computer
	
	if err != nil { // Log error catching
      log.Fatal(err)
	}
	
    defer outFile.Close()
	jpeg.Encode(outFile, imgSet, nil)
	
}


///// Converts original photo using negative filter /////
func Negative(){

	file, err := os.Open("Original.jpg") // OS opens file
	img, err := jpeg.Decode(file) // Turns img to img object

	b := img.Bounds() // Bounds are to know the size of the image so new object can be created]

	imgSet := image.NewRGBA(b) // Instantiate new img object

	// Loop takes old RGB, subtracts from RGB maximum value, and sets new RGB values
	for y := 0; y < b.Max.Y; y++ {
		for x := 0; x < b.Max.X; x++ {
			oldPixel := img.At(x, y)
			r, g, b, a := oldPixel.RGBA() // Get old pixel's values
			// Subtract each old pixel value from the maximum (65535 not 255)
			r = 65535 - r
			g = 65535 - g
			b = 65535 - b
			a = 65535 - a
			// Shift 8 bits so most signigicant bits are at the bottom, these bits contain the color info
			// The color.RGBA requires uint8
			pixel := color.RGBA{uint8(r >> 8), uint8(g >> 8), uint8(b >> 8), uint8(a >> 8)}
			imgSet.Set(x, y, pixel)
		}
    }

	outFile, err := os.Create("Negative.jpg") // Save output img object to img file on computer
	
	if err != nil { // Log error catching
      log.Fatal(err)
	}
	
    defer outFile.Close()
	jpeg.Encode(outFile, imgSet, nil)
	
}

///// Adds sepia filter to original photo /////
func Sepia(){

	file, err := os.Open("Original.jpg") // OS opens file
	img, err := jpeg.Decode(file) // Turns img to img object

	b := img.Bounds() // Bounds are to know the size of the image so new object can be created

	imgSet := image.NewRGBA(b) // Instantiate new img object

	// Loop through each pixel
	for y := 0; y < b.Max.Y; y++ {
		for x := 0; x < b.Max.X; x++ {
			oldPixel := img.At(x, y) // Select pixel
			r, g, b, a := oldPixel.RGBA() // Get the RGBA values for that pixel

			// Apply sepia filter weights
			rNew := int(math.Round((float64(r) * .393) + (float64(g) * .769) + (float64(b) * .189)))
			gNew := int(math.Round((float64(r) * .349) + (float64(g) * .686) + (float64(b) * .168)))
			bNew := int(math.Round((float64(r) * .272) + (float64(g) * .534) + (float64(b) * .131)))
			
			// Prevent RGB value overflow 
			if rNew > 65535 {
				rNew = 65535
			}
			if gNew > 65535 {
				gNew = 65535
			}
			if bNew > 65535 {
				bNew = 65535
			}

			// Shift 8 bits so most signigicant bits are at the bottom
			pixel := color.RGBA{uint8(rNew >> 8), uint8(gNew >> 8), uint8(bNew >> 8), uint8(a >> 8)}
			imgSet.Set(x, y, pixel)
		}
    }

	outFile, err := os.Create("Sepia.jpg") // Save output img object to img file on computer
	
	if err != nil { // Log error catching
      log.Fatal(err)
	}
	
    defer outFile.Close()
	jpeg.Encode(outFile, imgSet, nil)
	
}


///// Spins off concurrent threads doing different filters /////
func main() {

	go BlackAndWhite()
	go Sepia()
	go Negative()

	// Main spins off thread and quits, so add wait for other thread to finish
	time.Sleep(10 * time.Second)
	
}