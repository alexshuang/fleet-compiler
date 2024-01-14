# ===- operator.py -------------------------------------------------------------
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

import re
import os

import fleet_compiler.frontend.python.operators.python.ops
import fleet_compiler.frontend.python.operators.python.time.ops
import fleet_compiler.frontend.python.operators.numpy.ops
import fleet_compiler.frontend.python.operators.numpy.random.ops


def get_function_names(file_path):
    content = open(file_path, 'r').read()
    pattern = re.compile(r'def\s+(\w+)\s*\(')
    matches = pattern.findall(content)
    return matches


def find_ops_paths(root_directory='operators', file_name='ops.py'):
    ops_paths = []
    current_directory = os.path.dirname(os.path.realpath(__file__))
    current_directory += f'/{root_directory}' 
    for root, dirs, files in os.walk(current_directory):
        if file_name in files:
            ops_paths.append(os.path.join(root, file_name))
    return ops_paths


class Operation:
    def __init__(self) -> None:
        self.op_tab = {}
        self.register_operators()

    def register_operators(self):
        current_directory = os.path.dirname(os.path.realpath(__file__))
        current_directory_parts = [o for o in current_directory.split('/') if o != '']
        clip_len = len(current_directory_parts) - 3 # fleet_compiler/frontend/python/...
        for file_path in find_ops_paths():
            func_names = get_function_names(file_path)
            path_parts = [o for o in file_path.split('/') if o != '']
            path_parts[-1] = path_parts[-1][:-3] # cut .py from ops.py
            for op in func_names:
                key = f"{'.'.join(path_parts[len(current_directory_parts)+1:-1])}.{op}"
                impl = eval(f"{'.'.join(path_parts[clip_len:])}.{op}")
                self.op_tab[key] = impl

    def lookup(self, key: str):
        return self.op_tab[key]
    
    def has(self, key: str):
        return key in self.op_tab
