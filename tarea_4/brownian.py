import pycuda.autoinit
import pycuda.driver as cuda
import pycuda.gpuarray as ga
import pycuda.curandom as curand
import numpy as np


def brownian(x0, n, dt, delta, out=None):

    x0 = np.asarray(x0)
    r = curand.MRG32k3aRandomNumberGenerator().gen_normal(x0.shape + (n,), np.float32).get()
    if out is None:
        out = np.empty(r.shape)

    np.cumsum(r, axis=-1, out=out)

    out += np.expand_dims(x0, axis=-1)

    return out
