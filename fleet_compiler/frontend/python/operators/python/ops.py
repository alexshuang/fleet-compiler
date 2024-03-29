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

def _print(args, kwargs):
    return print(*args, **kwargs)

def _assert(args, kwargs):
    assert(args[0])

def _sum(args, kwargs):
    return sum(*args, **kwargs)

def _slice(args, kwargs):
    slice_str = args[1]
    if ':' in slice_str:
        parts = [int(o) if o else None for o in slice_str.split(':')]
        return args[0][slice(*parts)]
    else:
        return args[0][eval(slice_str)]