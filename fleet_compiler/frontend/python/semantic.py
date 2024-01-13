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

from .syntax import *
from .scope import Scope
from .symbolic import *
from .pass_manager import *


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
    
    def visitFunctionDef(self, node: FunctionDef):
        self.scope.update(node.name, FunctionSymbol(SymbolKind.FunctionSymbol, node))
        self.enter()
        super().visitFunctionDef(node)
        self.exit()
    
    def visitParamentDef(self, node: ParamentDef):
        self.scope.update(node.name, VariableSymbol(SymbolKind.VariableSymbol, node))
        return super().visitParamentDef(node)

    def visitFunctionCall(self, node: FunctionCall):
        node.sym = self.scope.get(node.name)
        return super().visitFunctionCall(node)

    def visitVariableDef(self, node: VariableDef):
        self.scope.update(node.name, VariableSymbol(SymbolKind.VariableSymbol, node))
        return super().visitVariableDef(node)

    def visitVariable(self, node: Variable):
        node.sym = self.scope.get(node.name)
        return super().visitVariable(node)

    def enter(self):
        self.last_scope = self.scope
        self.scope = Scope(self.scope)
    
    def exit(self):
        self.scope = self.last_scope


class OperatorReferenceResolvePass(Pass):
    '''
    find the target built-in or numpy functions for unsolved function calls.
    '''
    def __init__(self) -> None:
        super().__init__()
        self.alias_tab = {}
        self.builtins = ['print', 'assert']

    def visitImportStatement(self, node: ImportStatement):
        if node.alias != "":
            self.alias_tab[node.alias] = node.package

    def visitFunctionCall(self, node: FunctionCall):
        # replace function name
        if node.sym is None:
            parts = node.name.split('.')
            pkg_or_func = parts[0]
            if pkg_or_func in self.builtins: # print -> python._print
                op_name = 'python.' + '.'.join(parts[:-1]) + "_" + parts[-1]
            elif pkg_or_func in self.alias_tab: # np.xxx -> numpy.xxx
                op_name = '.'.join([self.alias_tab[pkg_or_func]] + \
                    node.name.split('.')[1:])
            else:
                op_name = node.name
            node.sym = OperatorSymbol(SymbolKind.OperatorSymbol, op_name, self)
        return super().visitFunctionCall(node)
