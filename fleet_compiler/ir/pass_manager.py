# ===- pass_manager.py -------------------------------------------------------------
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
from .core import *
from .dialects.builtin import ModuleOp
from .pattern_rewriter import *


class Pass(ABC):
    name = ""

    def __init__(self) -> None:
        def camel_to_kebab(camel_case_string):
            return re.sub(r'([a-z])([A-Z])', r'\1-\2', camel_case_string).lower()

        if self.name == "":
            self.name = camel_to_kebab(self.__class__.__name__)

    def __call__(self, op: Operation) -> bool:
        return self.run_on_operation(op)

    @abstractmethod
    def run_on_operation(self, op: Operation) -> bool: ...


class PassManager:
    def __init__(self, dump_intermediates_to: str = None,
                 dump_ir_before_pass: bool = False,
                 dump_ir_after_pass: bool = False):
        self.passes = []
        self.dump_intermediates_to = dump_intermediates_to
        self.dump_ir_before_pass = dump_ir_before_pass
        self.dump_ir_after_pass = dump_ir_after_pass
    
    def add(self, p: Pass):
        if p not in self.passes:
            self.passes.append(p)

    def run(self, op: ModuleOp):
        for i, p in enumerate(self.passes):
            if self.dump_ir_before_pass:
                print(f"------\nBefore {p.name}:")
                op.dump()
                print(f"------")

            if self.dump_intermediates_to:
                fn = f'{self.dump_intermediates_to}/{i:03d}-before-{p.name}'
                with open(fn, 'w') as fp:
                    op.dump(fp)

            p(op)

            if self.dump_ir_after_pass:
                print(f"------\nAfter {p.name}:")
                op.dump()
                print(f"------")

            if self.dump_intermediates_to:
                fn = f'{self.dump_intermediates_to}/{i:03d}-after-{p.name}'
                with open(fn, 'w') as fp:
                    op.dump(fp)
