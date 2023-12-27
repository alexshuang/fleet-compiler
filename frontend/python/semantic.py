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

from syntax import *
from scope import Scope
from symbolic import *
from pass_manager import *


class ReferenceResolvePass(Pass):
    '''
    find the target function and variable definitions for the function call and variable.
    '''
    def __init__(self) -> None:
        super().__init__()
        self.scope:Scope = None
        self.last_scope:Scope = None

    def visitModule(self, node: AstModule):
        self.enter()
        return super().visitModule(node)
    
    def visitFunctionDecl(self, node: FunctionDecl):
        self.scope.update(node.name, FunctionSymbol(SymbolKind.FunctionSymbol, node))
        self.enter()
        super().visitFunctionDecl(node)
        self.exit()
    
    def visitParameterDecl(self, node: ParameterDecl):
        self.scope.update(node.name, VariableSymbol(SymbolKind.VariableSymbol, node))
        return super().visitParameterDecl(node)

    def visitFunctionCall(self, node: FunctionCall):
        node.sym = self.scope.get(node.name)
        return super().visitFunctionCall(node)

    def visitVariableDecl(self, node: VariableDecl):
        self.scope.update(node.name, VariableSymbol(SymbolKind.VariableSymbol, node))
        return super().visitVariableDecl(node)

    def visitVariable(self, node: Variable):
        node.sym = self.scope.get(node.name)
        return super().visitVariable(node)

    def enter(self):
        self.last_scope = self.scope
        self.scope = Scope(self.scope)
    
    def exit(self):
        self.scope = self.last_scope


class ReplaceAliasOperationNamePass(Pass):
    '''
    Replace the function name of the imported alias package with the original package.
    for example, replace 'np.matmul' with 'numpy.matmul' while 'import numpy as np'.
    '''
    def __init__(self) -> None:
        super().__init__()
        self.alias_tab = {}

    def visitImportStatement(self, node: ImportStatement):
        if node.alias != "":
            self.alias_tab[node.alias] = node.package

    def visitFunctionCall(self, node: FunctionCall):
        # replace function name
        if '.' in node.name:
            alias = node.name.split('.')[0]
            if alias in self.alias_tab:
                node.name = '.'.join([self.alias_tab[alias]] + \
                    node.name.split('.')[1:])
        return super().visitFunctionCall(node)
