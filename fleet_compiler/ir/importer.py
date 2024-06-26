
# ===- ir_gen.py -------------------------------------------------------------
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

from .core import *
from .builder import *
from .dialects.builtin import *
from .dialects import arith

from fleet_compiler.frontend.python.ast import (
    AstModule,
    AstVisitor,
    Block as ASTBlock,
    VariableDef,
    IntegerLiteral,
)

class ConvertASTtoMLIR(AstVisitor):
    ast: AstModule
    module: ModuleOp

    def __init__(self, ast: AstModule) -> None:
        super().__init__()
        self.ast = ast

    def convert(self):
        return self.visitModule(self.ast)

    def visitModule(self, node: AstModule):
        self.module = ModuleOp()
        self.visitBlock(node.block, self.module.body.blocks[0])
        return self.module

    def visitBlock(self, ast_block: ASTBlock, block: Block):
        with Builder.at_end(block):
            for o in ast_block.stmts:
                self.visit(o)

    def visitVariableDef(self, node: VariableDef):
        # init_op = self.visit(node.init)
        # self.sym_table[node.name] = init_op
        self.visit(node.init)

    def visitIntegerLiteral(self, node: IntegerLiteral):
        type = IntegerType(32, True)
        attr = IntegerAttr(node.value, type)
        op = arith.Constant(attr, type)
        builder = ImplicitBuilder().get()
        builder.insert(op)


class ASTModuleImporter():
    def __init__(self, module: AstModule) -> None:
        self.module = module
    
    def import_graph(self):
        m = ConvertASTtoMLIR(self.module).convert()
        return m
    