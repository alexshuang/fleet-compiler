
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
import unicodedata
from itertools import tee
import functools

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
        self.parts = iter(hint.split('_'))

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
        assert (data := self._get()) in ["device", "python"]
        return 'accelerator' if data == 'device' else 'host'

    def _parse_name(self):
        return self._get()

    def _parse_sig(self):
        assert self._get() == 'sig'
        in_abi = out_abi = None
        if self._peak() != 'ins':
            in_abi = self._get()
        if self._peak() != 'ins':
            out_abi = self._get()
        return [in_abi, out_abi]

    def _parse_input_types(self):
        assert self._get() == 'ins'
        types = []
        data = self._peak()
        while data and data != 'outs':
            types.append(self._get())
            data = self._peak() 
        return types

    def _parse_output_types(self):
        def is_valid(data: str):
            substr = ['f32', 'i32', 'f64', 'i64']
            for s in substr:
                if s in data:
                    return True
            return False

        if not self._peak():
            return []

        assert self._get() == 'outs'
        types = []
        data = self._peak()
        while data and is_valid(data):
            types.append(self._get())
            data = self._peak()
        return types

    def _parse_attributes(self):
        def is_number(s):
            try:
                float(s)
                return True
            except ValueError:
                pass

            try:
                unicodedata.numeric(s)
                return True
            except (TypeError, ValueError):
                pass
            return False

        res = {}
        while self._peak():
            assert(not is_number(key := self._get()))
            vals = []
            while is_number(self._peak()):
                vals.append(eval(self._get()))
            res[key] = vals[0] if len(vals) == 1 else vals

        return res


def get_dispatch_function(info: DispatchInfo):
    kwargs = info.attrs
    if info.target == 'python':
        func = python_get_dispatch_function(info.name)
    return functools.partial(func, **kwargs)
    

def python_get_dispatch_function(name: str):
    python_dispatch_functions = {
        "add": np.add,
        "sub": np.subtract,
        "mul": np.multiply,
        "print": print,
    }
    return python_dispatch_functions[name]
