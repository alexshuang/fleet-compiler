
# ===- dispatch.py -------------------------------------------------------------
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

from __future__ import annotations
from dataclasses import dataclass
from itertools import tee
import functools
import re

from fleet_compiler.ir.core import *
from fleet_compiler.ir.dialects.builtin import *


@dataclass
class DispatchInfo:
    type: str
    target: str
    name: str
    sig: list[str]
    in_types: list[str]
    out_types: list[str]
    attrs: dict[str, list|Any]


class DispatchHintParser:
    def __init__(self, hint: str) -> None:
        self.hint = hint
        self.parts = iter(hint.split('_'))
        self.num_args = 0
        self.num_outs = 0

    def _get(self):
        return next(self.parts, None)

    def _peak(self):
        new, self.parts = tee(self.parts)
        return next(new, None)

    def parse(self):
        type = self._parse_type()
        name = self._parse_name()
        sig = self._parse_sig()
        in_types = self._parse_input_types()
        out_types = self._parse_output_types()
        attrs = self._parse_attributes()
        return DispatchInfo(type, '', name, sig, in_types, out_types, attrs)
    
    def _parse_type(self):
        assert (data := self._get()) in ["device", "python"], f"invalid function hint: {self.hint}"
        return 'accelerator' if data == 'device' else 'host'

    def _parse_name(self):
        res = []
        while self._peak() != 'sig':
            res.append(self._get())
        return '_'.join(res) if len(res) > 1 else res[0]

    def _parse_sig(self):
        assert self._get() == 'sig', f"invalid function hint: {self.hint}"
        in_abi = out_abi = None
        if self._peak() != 'ins':
            in_abi = self._get()
        if self._peak() != 'ins':
            out_abi = self._get()
        self.num_args = len(in_abi) if in_abi else 0
        self.num_outs = len(out_abi) if out_abi else 0
        return [in_abi, out_abi]

    def _parse_input_types(self):
        assert self._get() == 'ins', f"invalid function hint: {self.hint}"
        types = []
        for _ in range(self.num_args):
            assert (data := self._get()) != 'outs'
            types.append(data)
        return types

    def _parse_output_types(self):
        if self.num_outs == 0:
            return []

        assert self._get() == 'outs', f"invalid function hint: {self.hint}"
        types = []
        for _ in range(self.num_outs):
            types.append(self._get())
        return types

    def _parse_attributes(self):
        def is_number(s):
            return bool(re.match(r'^-?\d+(\.\d+)?$', s))
        
        def is_value(s):
            return is_number(s) or s == 'True' or s == 'False'

        res = {}
        while self._peak():
            name = []
            while (val := self._peak()) and not is_value(val):
                name.append(self._get())
            key = '_'.join(name)
            vals = []
            while (val := self._peak()) and is_value(val):
                vals.append(eval(self._get()))
            res[key] = vals[0] if len(vals) == 1 else vals
        return res


def get_dispatch_function(info: DispatchInfo):
    assert info.target == 'python'
    func = python_get_dispatch_function(info)
    kwargs = info.attrs
    return functools.partial(func, **kwargs)
    

def ir_type_to_py_type(raw: str):
    parts = raw.split('x')
    shape = [eval(o) for o in parts[:-1]]
    dtype = parts[-1]
    if dtype == 'f32':
        dtype = np.float32
    elif dtype == 'f64':
        dtype = np.float64
    elif dtype == 'i32':
        dtype = np.int32
    elif dtype == 'i64':
        dtype = np.int64
    else:
        raise ValueError(f"Invalid dtype {dtype}")
    return shape, dtype


def splat_func(info: DispatchInfo):
    def _splat_func(x):
        shape, dtype = ir_type_to_py_type(info.out_types[0])
        return np.full(shape, x, dtype=dtype)
    return _splat_func


def slice_func(x, **kwargs):
    start = kwargs['start']
    size = kwargs['size']
    slices = tuple(slice(start[i], start[i] + size[i]) for i in range(len(start)))
    return x[slices]


def concat_func(*args, **kwargs):
    return np.concatenate(args, **kwargs)


def reshape_func(x, **kwargs):
    return x.reshape(kwargs['new_shape'])


def python_get_dispatch_function(info: DispatchInfo):
    dispatch_functions = {
        "add": np.add,
        "sub": np.subtract,
        "mul": np.multiply,
        "div": np.divide,
        "gather": lambda array, idx: array[idx].squeeze(),
        "transpose": np.transpose,
        "matmul": np.matmul,
        "mean": np.mean,
        "var": np.var,
        "sqrt": np.sqrt,
        "pow": np.power,
        "tanh": np.tanh,
        "max": np.max,
        "exp": np.exp,
        "slice": slice_func,
        "concat": concat_func,
        "reshape": reshape_func,
        "reduce_sum": np.sum,
        "reduce_max": np.max,
        "cast": lambda x: x,
        "splat": splat_func(info),
        "print": print,
    }
    return dispatch_functions[info.name]
