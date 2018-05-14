import pycuda.autoinit
import pycuda.driver as cuda
import pycuda.gpuarray as gpuarray
from pycuda.reduction import ReductionKernel
from pycuda.curandom import rand as curand
import numpy

a = curand((1000*1000), dtype=numpy.float32)
b = curand((1000*1000), dtype=numpy.float32)

piKernel = ReductionKernel(numpy.float32, neutral="0",
		reduce_expr="a+b", map_expr="float(x[i] * x[i] + y[i] * y[i]) <= 1.0f",
		arguments="float *x, float*y")

pi = (4.0 * piKernel(a, b).get()) / (1000*1000)

print(pi)
