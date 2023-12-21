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
        for o in node.args:
            print(self.prefix + f"    {self.visit(o)}")

    def visitVariable(self, node: Variable):
        ref_str = "(resolved)" if node.sym else "(not resolved)"
        return f'arg {node.name} {ref_str}'


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
    
    def visitFunctionCall(self, node: FunctionCall):
        node.sym = self.scope.get(node.name)
        for o in node.args:
            self.visit(o)
    
    def visitVariableDecl(self, node: VariableDecl):
        self.scope.update(node.name, VariableSymbol(SymbolKind.VariableSymbol, node))
    
    def visitVariable(self, node: Variable):
        node.sym = self.scope.get(node.name)
    
    def visitDecimalLiteral(self, node: DecimalLiteral):
        return super().visitDecimalLiteral(node)
    
    def visitEmptyStatement(self, node: EmptyStatement):
        return super().visitEmptyStatement(node)
    
    def visitExpressionStatement(self, node: ExpressionStatement):
        return super().visitExpressionStatement(node)
    
    def visitIntegerLiteral(self, node: IntegerLiteral):
        return super().visitIntegerLiteral(node)

    def visitNoneLiteral(self, node: NoneLiteral):
        return super().visitNoneLiteral(node)
    
    def visitStringLiteral(self, node: StringLiteral):
        return super().visitStringLiteral(node)
    
    def enter(self):
        self.last_scope = self.scope
        self.scope = Scope(self.scope)
    
    def exit(self):
        self.scope = self.last_scope
