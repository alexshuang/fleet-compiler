# ===- ops.py -------------------------------------------------------------
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# ===---------------------------------------------------------------------------

import numpy as np

def matmul(args, kwargs):
    return np.matmul(*args, **kwargs)

def sqrt(args, kwargs):
    return np.sqrt(*args, **kwargs)

def tanh(args, kwargs):
    return np.tanh(*args, **kwargs)

def shape(args, kwargs):
    return np.shape(*args, **kwargs)

def power(args, kwargs):
    return np.power(*args, **kwargs)

def ones(args, kwargs):
    return np.ones(*args, **kwargs)

def sum(args, kwargs):
    return np.sum(*args, **kwargs)

def allclose(args, kwargs):
    return np.allclose(*args, **kwargs)

def arange(args, kwargs):
    return np.arange(*args, **kwargs)

def stack(args, kwargs):
    return np.stack(*args, **kwargs)

def mean(args, kwargs):
    return np.mean(*args, **kwargs)

def var(args, kwargs):
    return np.var(*args, **kwargs)

def split(args, kwargs):
    return np.split(*args, **kwargs)

def tri(args, kwargs):
    return np.tri(*args, **kwargs)

def transpose(args, kwargs):
    return np.transpose(*args, **kwargs)

def max(args, kwargs):
    return np.max(*args, **kwargs)

def exp(args, kwargs):
    return np.exp(*args, **kwargs)

def hstack(args, kwargs):
    return np.hstack(*args, **kwargs)
