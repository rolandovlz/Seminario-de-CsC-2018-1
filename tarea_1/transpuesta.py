import pycuda.autoinit
import pycuda.gpuarray as gpuarray
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
from pycuda.curandom import rand as curand
import numpy

n = 1024
m = 1024
TILE_DIM = 32

transpuestaKernel = """
__global__ void transpuesta(float *a, float *b) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int i = y + x * %(EME)s;
	int j = x + y * %(ENE)s;

	if(i < (%(ENE)s * %(EME)s))
		b[j] = a[i];
}
"""
transpuestaKernel = transpuestaKernel % {
		"ENE" : n,
		"EME" : m
}

a_gpu = curand((n*m))
b_gpu = gpuarray.empty_like(a_gpu)

mod = SourceModule(transpuestaKernel)
func = mod.get_function("transpuesta")

func(
		a_gpu,
		b_gpu,
		block=(TILE_DIM, TILE_DIM, 1),
		grid=(m//TILE_DIM, n//TILE_DIM, 1))

a_gpu = a_gpu.reshape((n,m))
b_gpu = b_gpu.reshape((m,n))

print(a_gpu)
print("-" * 80)
print(b_gpu)


