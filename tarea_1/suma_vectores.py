import pycuda.autoinit
import pycuda.driver as cuda
import numpy
from pycuda.compiler import SourceModule

n = 500

a,b = numpy.random.randn(n), numpy.random.randn(n)
a,b = a.astype(numpy.float32), b.astype(numpy.float32)
c   = numpy.zeros_like(a)

mod = SourceModule("""
__global__ void kernel_suma (float *a, float *b, float *c) {

	int i = threadIdx.x;

	c[i] = a[i] + b[i];
}
""")

func = mod.get_function("kernel_suma")
func(cuda.In(a), cuda.In(a), cuda.Out(c), block=(n,1,1), grid=(1,1))

print(c)
