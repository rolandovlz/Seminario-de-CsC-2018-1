import pycuda.autoinit
import pycuda.gpuarray as gpuarray
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
from pycuda.curandom import rand as curand
import numpy

n = 1024

matMultKernel = """
__global__ void mat_mult(float *a, float *b, float *c) {
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

	for(int k = 0; k < %(ENE)s; k++)
		c[y + x * %(ENE)s] += a[k + x * %(ENE)s] * b[y + k * %(ENE)s];
}
"""

a_gpu = curand((n,n))
b_gpu = curand((n,n))
c_gpu = gpuarray.zeros((n,n), dtype=numpy.float32)

matMultKernel = matMultKernel % {
		"ENE" : n,
}

mod = SourceModule(matMultKernel)
mat_mult = mod.get_function("mat_mult")

mat_mult(
		a_gpu, b_gpu,
		c_gpu,
		block=(32, 32, 1),
		grid=(n//32, n//32, 1)
		)

print(a_gpu.get())
print("-" * 80)
print(b_gpu.get())
print("-" * 80)
print(c_gpu.get())
