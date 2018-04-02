import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy as np
from scipy.misc import imsave, imread

blurKernel = """
// Necesitamos un pixVal por cada canal que tenga la imagen.
// Como normalmente tenemos 3 lo dejare asi.
typedef struct pixVal {
    int r;
    int g;
    int b;
} pixVal;
    
__global__ void blurImage(float *input, float *output) {
    int col = threadIdx.x + blockIdx.x * blockDim.x;
    int row = threadIdx.y + blockIdx.y * blockDim.y;

    if (col < %(WIDTH)s && row < %(HEIGHT)s) {
        pixVal pv;
        int pixels = 0;

        pv.r = 0;
        pv.g = 0;
        pv.b = 0;

        for (int blurRow = -%(BLUR_S)s; blurRow < %(BLUR_S)s + 1; ++blurRow) {
            for (int blurCol = -%(BLUR_S)s; blurCol < %(BLUR_S)s + 1; ++blurCol) {
                int curRow = row + blurRow * %(CHANNELS)s;
                int curCol = col + blurCol * %(CHANNELS)s;

                if (curRow > -1 && curRow < %(HEIGHT)s && \
                        curCol > -1 && curCol < %(WIDTH)s ) {
                        pv.r += input[(curRow*%(WIDTH)s+curCol)*%(CHANNELS)s];
                        pv.g += input[(curRow*%(WIDTH)s+curCol)*%(CHANNELS)s + 1];
                        pv.b += input[(curRow*%(WIDTH)s+curCol)*%(CHANNELS)s + 2];
                        pixels++;
                }
            }
        }
    output[(row * %(WIDTH)s + col) * %(CHANNELS)s] = pv.r/pixels;
    output[(row * %(WIDTH)s + col) * %(CHANNELS)s + 1] = pv.g/pixels;
    output[(row * %(WIDTH)s + col) * %(CHANNELS)s + 2] = pv.b/pixels;
    }
}
"""
imageIn = imread("test.jpg").astype(np.float32)
imageOut = np.copy(imageIn).astype(np.float32)

WIDTH, HEIGHT, CHANNELS = imageIn.shape
BLUR_S = 2
blurKernel = blurKernel % {
        "HEIGHT": HEIGHT,
        "WIDTH"   : WIDTH,
        "CHANNELS": CHANNELS,
        "BLUR_S"  : BLUR_S
}

mod = SourceModule(blurKernel)
func = mod.get_function("blurImage")

func(
        cuda.In(imageIn),
	cuda.Out(imageOut),
	block = (32,32,1),
	grid = (int((WIDTH+32)//32), int((HEIGHT+32)//32), 1)
)

imsave("output_blur.jpg", imageOut.reshape(WIDTH,HEIGHT,CHANNELS))
