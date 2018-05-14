import pycuda.autoinit
import pycuda.driver as cuda
import pycuda.gpuarray as gpuarray
from pycuda.reduction import ReductionKernel
from pycuda.curandom import rand as curand
import numpy

n = 1000000
a = curand(n, dtype=numpy.float32)
b = curand(n, dtype=numpy.float32)

dotKernel = ReductionKernel(numpy.float32, neutral="0",
		reduce_expr="a+b", map_expr="x[i]*y[i]",
		arguments="float *x, float*y")

doot = dotKernel(a, b).get()

print(doot)


