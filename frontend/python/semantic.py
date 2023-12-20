# ===- _ast.py -------------------------------------------------------------
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

from syntax import *


class AstDumper(AstVisitor):
    def __init__(self, prefix="") -> None:
        super().__init__()
        self.prefix = prefix
    
    def visitModule(self, node: AstModule):
        print(self.prefix + "Module:")
        self.inc_indent()
        self.visitBlock(node.block)
        self.dec_indent()
    
    def visitBlock(self, node: Block):
        print(self.prefix + "Block:")
        self.inc_indent()
        for o in node.stmts:
            self.visit(o)
        self.dec_indent()

    def visitFunctionDecl(self, node: FunctionDecl):
        print(self.prefix + f"Function Decl {node.name}")
        self.inc_indent()
        self.visitBlock(node.block)
        self.dec_indent()
    
    def visitFunctionCall(self, node: FunctionCall):
        print(self.prefix + f"Function Call {node.name}, args: {node.args}")
    
    def visitReturnStatement(self, node: ReturnStatement):
        print(self.prefix + f"Return {node.ret}")

    def inc_indent(self):
        self.prefix += "  "

    def dec_indent(self):
        self.prefix = self.prefix[:-2]
