import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy as np
from scipy.misc import imsave, imread

grayKernel = """
	__global__ void colorConvert(float *rgbImage, float *grayImage){
 		int x = threadIdx.x + blockIdx.x * blockDim.x;
  		int y = threadIdx.y + blockIdx.y * blockDim.y;
		
		if(x < %(WIDTH)s && y < %(HEIGHT)s){
			// get 1D coordinate for the grayscale image
			int grayOffset = y * %(WIDTH)s + x;
			// one can think of the RGB image having
			// CHANNEL times columns than gray scale image
			int rgbOffset  = grayOffset * %(CHANNELS)s;
    		float r = rgbImage[rgbOffset];		// red value for pixel
    		float g = rgbImage[rgbOffset + 1];	// green value for pixel
			float b = rgbImage[rgbOffset + 2];	// blue value for pixel
			
			float grayVal = 0.21f * r + 0.71f * g + 0.07f * b;
			grayImage[rgbOffset + 0] = grayVal;
			grayImage[rgbOffset + 1] = grayVal;
			grayImage[rgbOffset + 2] = grayVal;
  		}
	}
"""

imageIn = imread("test.jpg").astype(np.float32)
imageOut = np.copy(imageIn).astype(np.float32)

WIDTH, HEIGHT, CHANNELS = imageIn.shape

grayKernel = grayKernel % {
    "HEIGHT"  : HEIGHT,
    "WIDTH"   : WIDTH,
    
	"CHANNELS": CHANNELS
    }

mod = SourceModule(grayKernel)
func = mod.get_function("colorConvert")

func(
		cuda.In(imageIn),
		cuda.Out(imageOut),
		block = (32,32,1),
		grid = (int((WIDTH+32)//32), int((HEIGHT+32)//32), 1)
		)

imsave("output_rgb2g.jpg", imageOut.reshape(WIDTH,HEIGHT,CHANNELS))
