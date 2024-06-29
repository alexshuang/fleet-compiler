
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

from fleet_compiler.frontend.ast import (
    AstNode,
    AstModule,
    AstVisitor,
    Block as ASTBlock,
    VariableDef,
    IntegerLiteral,
    DecimalLiteral,
    BooleanLiteral,
    NoneLiteral,
    ListContent
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
        self.create(arith.Constant(attr, type))

    def visitDecimalLiteral(self, node: DecimalLiteral):
        type = FloatType(32)
        attr = FloatAttr(node.value, type)
        self.create(arith.Constant(attr, type))

    def visitBooleanLiteral(self, node: BooleanLiteral):
        type = BoolType()
        attr = BoolAttr(node.value)
        self.create(arith.Constant(attr, type))

    def visitNoneLiteral(self, node: NoneLiteral):
        type = NoneType()
        attr = NoneAttr()
        self.create(arith.Constant(attr, type))
    
    def visitListContent(self, node: ListContent):
        support_elem_types = [BooleanLiteral, IntegerLiteral, DecimalLiteral]
        elem_type_priority = {o:i for i, o in enumerate(support_elem_types)}
        dtype_funcs = [bool, int, float]

        def is_tensor_or_vector(node: ListContent):
            if not node.exps or len(node.exps) == 0:
                return False
            for o in node.exps:
                if type(o.exp) not in support_elem_types:
                    return False
            return True

        def get_dense_attr(node: ListContent):
            priority = -1
            data = []
            for o in node.exps:
                elem = o.exp
                data.append(elem.value)
                priority = max(elem_type_priority[type(elem)], priority)
            dtype_fn = dtype_funcs[priority]
            # 1-D array support now
            data = list(map(dtype_fn, data))
            elem_type = ConvertASTtoMLIR.create_ir_type_from_literal(support_elem_types[priority])
            tensor_type = RankedTensorType([len(data)], elem_type)
            return DenseIntOrFPElementsAttr(data, tensor_type)

        if is_tensor_or_vector(node):
            dense_attr = get_dense_attr(node)
            self.create(arith.Constant(dense_attr, dense_attr.type))
        else:
            for o in node.exps:
                self.visit(o)

    def create(self, op: Operation):
        builder = ImplicitBuilder().get()
        builder.insert(op)
    
    @staticmethod
    def create_ir_type_from_literal(literal: AstNode):
        ir_type_factory = {
            IntegerLiteral: IntegerType(32, True),
            DecimalLiteral: FloatType(32),
            BooleanLiteral: BoolType(),
            NoneLiteral: NoneType(),
        }
        return ir_type_factory[literal]


class ASTModuleImporter():
    def __init__(self, module: AstModule) -> None:
        self.module = module
    
    def import_graph(self):
        m = ConvertASTtoMLIR(self.module).convert()
        return m
    