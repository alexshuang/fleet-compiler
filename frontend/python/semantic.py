# ===- semantic.py -------------------------------------------------------------
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
# Semantic analysis:
#   Reference resolution:
#       RefVisitor() used to find the target function for the function call.
# 
# ===---------------------------------------------------------------------------

from scope import Scope
from symbolic import *
from syntax import *


semantic_anlyisys_pipeline = []


class RefDumper(AstDumper):
    def visitFunctionCall(self, node: FunctionCall):
        ref_str = "(resolved)" if node.sym else "(not resolved)"
        print(self.prefix + f"Function Call {node.name}, args: {node.args}  {ref_str}")


class RefVisitor(AstVisitor):
    def __init__(self) -> None:
        super().__init__()
        self.scope:Scope = None
        self.last_scope:Scope = None

    def visitModule(self, node: AstModule):
        return super().visitModule(node)
    
    def visitReturnStatement(self, node: ReturnStatement):
        return super().visitReturnStatement(node)

    def visitBlock(self, node: Block):
        self.enter()
        ret = super().visitBlock(node)
        self.exit()
        return ret
    
    def visitFunctionDecl(self, node: FunctionDecl):
        self.scope.update(node.name, FunctionSymbol(SymbolKind.FunctionSymbol, node))
        return super().visitFunctionDecl(node)
    
    def visitFunctionCall(self, node: FunctionCall):
        node.sym = self.scope.get(node.name)
        return super().visitFunctionCall(node)
    
    def enter(self):
        self.last_scope = self.scope
        self.scope = Scope(self.scope)
    
    def exit(self):
        self.scope = self.last_scope
