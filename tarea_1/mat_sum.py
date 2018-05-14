import pycuda.autoinit
import pycuda.gpuarray as gpuarray
from pycuda.elementwise import ElementwiseKernel
from pycuda.curandom import rand as curand

n = 500
m = 400

sumaMat = ElementwiseKernel(
		"float *a, float *b, float *c",
		"c[i] = a[i] + b[i]",
		"add")

a_gpu = curand((n,m))
b_gpu = curand((n,m))

c_gpu = gpuarray.empty_like(a_gpu)
sumaMat(a_gpu, b_gpu, c_gpu)

print(a_gpu)
print(b_gpu)
print(c_gpu)
