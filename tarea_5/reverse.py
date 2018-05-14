import pycuda.autoinit
import pycuda.gpuarray as gpuarray
from pycuda.elementwise import ElementwiseKernel
from pycuda.curandom import rand as curand

n = 1000000

reverseKernel = ElementwiseKernel(
		"float *a, float *b, int c",
		"b[i] = a[n-1-i]",
		"reverse")

a_gpu = curand((n))
b_gpu = gpuarray.empty_like(a_gpu)

reverseKernel(a_gpu, b_gpu, n)

print(a_gpu)
print("-" * 80)
print(b_gpu)
print("-" * 80)
print(n)
