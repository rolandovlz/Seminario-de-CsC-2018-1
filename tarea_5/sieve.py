import pycuda.autoinit
import pycuda.gpuarray as gpuarray
from pycuda.elementwise import ElementwiseKernel
from pycuda.curandom import rand as curand
import numpy

n = 1000

sieveKernel = ElementwiseKernel(
		"int *a",
		"if(i > 1){for(int j = 2; i * j < n; j++){a[i*j] = 0;}}",
		"sieve")

a_gpu = gpuarray.arange(n, dtype=numpy.int32)
#print(a_gpu)

sieveKernel(a_gpu)

#print("-" * 80)
#print(a_gpu)
#print("-" * 80)
print('[ ', end="")
for i in a_gpu.get():
	if i > 1:
		print(i, end=" ")
print("\b]")
