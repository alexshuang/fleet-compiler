# ===- scope.py -------------------------------------------------------------
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

from .syntax import *
from .symbolic import *


class Scope:
    def __init__(self, parent=None) -> None:
        self.sym_table = {}
        self.parent = parent # parent scope
    
    def update(self, name: str, sym: Symbol):
        self.sym_table[name] = sym
    
    def get(self, name: str):
        if name in self.sym_table:
            return self.sym_table[name]
        else:
            if self.parent:
                return self.parent.get(name)
        return None
