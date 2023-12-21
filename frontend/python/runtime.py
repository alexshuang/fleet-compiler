# ===- runtime.py -------------------------------------------------------------
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

from syntax import AstVisitor, AstModule, Block, FunctionCall, FunctionDecl, ReturnStatement


class StackFrame:
    def __init__(self) -> None:
        self.variables = {}
    
    def update(self, name: str, value):
        self.variables[name] = value

    def get(self, name: str):
        return self.variables[name] if name in self.variables else None


class Interpreter(AstVisitor):
    def __init__(self) -> None:
        super().__init__()
        self.call_stack = []
    
    def visitModule(self, node: AstModule):
        return super().visitModule(node)

    def visitBlock(self, node: Block):
        def should_run(node):
            # Instructions that should be executed
            return isinstance(node, FunctionCall)

        ret = None
        self.enter()
        for o in node.stmts:
            if should_run(o):
                ret = self.visit(o)
                if ret == "return":
                    break
        self.exit()
        return ret
    
    def visitFunctionCall(self, node: FunctionCall):
        if node.sym:
            return self.visit(node.sym.node)
        else:
            if node.name == "print":
                return print(node.args)
            else:
                raise ValueError(f"Interpreter: Unsupport instruction {node.name}")

    def visitReturnStatement(self, node: ReturnStatement):
        return super().visitReturnStatement(node)

    def visitFunctionDecl(self, node: FunctionDecl):
        return super().visitFunctionDecl(node)
    
    def enter(self):
        self.call_stack.append(StackFrame())
    
    def exit(self):
        self.call_stack.pop()
