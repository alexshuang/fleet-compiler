# ===- pass.py -------------------------------------------------------------
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
from .ast import *


def camel_to_snake(name):
    snake_case = re.sub('([a-z0-9])([A-Z])', r'\1-\2', name).lower()
    return snake_case


class Pass(AstVisitor):
    def __init__(self) -> None:
        super().__init__()

    def run(self, node: AstModule, dump: bool = False):
        pass_name = camel_to_snake(self.__class__.__name__)
        if dump:
            dumper = AstDumper()
            print(f"\nBefore {pass_name}:")
            dumper.visit(node)
        ret = self.visitModule(node)
        if dump:
            print(f"\nAfter {pass_name}:")
            dumper.visit(node)
        return ret


class Pipeline:
    def __init__(self, name: str = ""):
        self.passes = []
        self.name = camel_to_snake(self.__class__.__name__)
    
    def add(self, new_pass: Pass):
        self.passes.append(new_pass)
    
    def run(self, node: AstModule, dump: bool = False):
        print(f"\nRun {self.name}:")
        for p in self.passes:
            p.run(node, dump)
