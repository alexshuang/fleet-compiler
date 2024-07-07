
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

import numpy as np

from .core import *
from .builder import *
from .dialects.builtin import *
from .dialects.func import *
from .dialects import arith, tosa, numpy as numpy_dialect, func

from fleet_compiler.frontend.lexer import (
    Op as opcode
)

from fleet_compiler.frontend.symbolic import (
    OperatorSymbol,
    FunctionSymbol
)

from fleet_compiler.frontend.ast import (
    AstNode,
    AstModule,
    AstVisitor,
    Block as ASTBlock,
    Variable,
    VariableDef,
    IntegerLiteral,
    DecimalLiteral,
    BooleanLiteral,
    NoneLiteral,
    ListContent,
    Binary,
    Unary,
    FunctionCall,
    ArgumentList,
    FunctionDef,
    ReturnStatement,
    BlockEnd,
    Signature,
    ParameterList,
    ParamentDef
)


def dtype_to_irtype(dtype):
    if dtype == np.int:
        return IntegerType
    elif dtype == np.float:
        return FloatType
    elif dtype == np.bool:
        return BoolType
    else:
        raise ValueError(f"unsupported dtype: {dtype}")


class ConvertASTtoMLIR(AstVisitor):
    ast: AstModule
    module: ModuleOp

    def __init__(self, ast: AstModule) -> None:
        super().__init__()
        self.ast = ast
        self.has_explicit_ret = False

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
        init = self.visit(node.init)
        builder = ImplicitBuilder().get()
        builder.symbol_table[node.name] = init

    def visitIntegerLiteral(self, node: IntegerLiteral):
        type = IntegerType(32, True)
        attr = IntegerAttr(node.value, type)
        return self.create(arith.ConstantOp(attr, type)).results[0]

    def visitDecimalLiteral(self, node: DecimalLiteral):
        type = FloatType(32)
        attr = FloatAttr(node.value, type)
        return self.create(arith.ConstantOp(attr, type)).results[0]

    def visitBooleanLiteral(self, node: BooleanLiteral):
        type = BoolType()
        attr = BoolAttr(node.value)
        return self.create(arith.ConstantOp(attr, type)).results[0]

    def visitNoneLiteral(self, node: NoneLiteral):
        type = NoneType()
        attr = NoneAttr()
        return self.create(arith.ConstantOp(attr, type)).results[0]
    
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
            return self.create(arith.ConstantOp(dense_attr, dense_attr.type)).results[0]
        else:
            return [self.visit(o) for o in node.exps]

    def visitVariable(self, node: Variable):
        value, _ = ImplicitBuilder().lookup_symbol(node.name)
        return value

    def visitBinary(self, node: Binary):
        if node.op == opcode.Assign:
            return self.visit(node.exp2)
        elif node.op == opcode.Plus:
            return self.make_add_op(node.exp1, node.exp2)
        elif node.op == opcode.Minus:
            return self.make_sub_op(node.exp1, node.exp2)
        elif node.op == opcode.Multiply:
            return self.make_mul_op(node.exp1, node.exp2)
        elif node.op == opcode.Divide:
            return self.make_div_op(node.exp1, node.exp2)

    def visitArgumentList(self, node: ArgumentList):
        args = [self.visit(o) for o in node.args if o]
        return args if len(args) > 0 else ""

    def visitFunctionCall(self, node: FunctionCall):
        arg_list = self.visitArgumentList(node.arg_list)
        args, kwargs = [], {}
        for o in arg_list:
            if isinstance(o, dict):
                kwargs.update(o)
            else:
                args.append(o)
        sym = node.sym
        if isinstance(sym, OperatorSymbol):
            if sym.op_name.startswith('numpy.'):
                return self.make_numpy_op(sym.op_name, *args, **kwargs)
        elif isinstance(sym, FunctionSymbol):
            func_op, _ = ImplicitBuilder().lookup_symbol(sym.node.name)
            return self.make_call_op(func_op, *args, **kwargs)

    def visitFunctionDef(self, node: FunctionDef):
        # self.visitSignature(node.signature)
        func_type = FunctionType([], [])
        func_op = self.create(func.FuncOp(node.name, func_type, []))
        self.visitBlock(node.block, func_op.regions[0].blocks[0])
        builder = ImplicitBuilder().get()
        builder.symbol_table[node.name] = func_op

    # def visitSignature(self, node: Signature):
    #     params = self.visitParameterList(node.param_list)

    # def visitParameterList(self, node: ParameterList):
    #     return [self.visitParamentDef(p) for p in node.params]

    # def visitParamentDef(self, node: ParamentDef):
    #     import pdb; pdb.set_trace()
    #     value, _ = ImplicitBuilder().lookup_symbol(node.name)
    #     return value

    def visitReturnStatement(self, node: ReturnStatement):
        ret_val = self.visit(node.ret)
        if not isinstance(ret_val, list):
            ret_val = [ret_val]
        ret_op = self.create(ReturnOp(ret_val))
        self.has_explicit_ret = True
        func_op = ret_op.parent.parent.parent
        func_op.function_type.output_types = [o.type for o in ret_val]

    def visitBlockEnd(self, node: BlockEnd):
        if not self.has_explicit_ret:
            self.create(ReturnOp([]))
            self.has_explicit_ret = False

    def create(self, op: Operation):
        builder = ImplicitBuilder().get()
        builder.insert(op)
        return op

    def make_add_op(self, node1: Unary, node2: Unary):
        lhs = self.visit(node1)
        rhs = self.visit(node2)
        if lhs.type != rhs.type:
            raise ValueError(f"lhs.type != rhs.type: {lhs.type} vs. {rhs.type}")
        if isinstance(lhs.type, IntegerType | BoolType):
            return self.create(arith.AddIOp(lhs, rhs)).results[0]
        elif isinstance(lhs.type, FloatType):
            return self.create(arith.AddFOp(lhs, rhs)).results[0]
        else:
            return self.create(tosa.AddOp(lhs, rhs)).results[0]

    def make_sub_op(self, node1: Unary, node2: Unary):
        lhs = self.visit(node1)
        rhs = self.visit(node2)
        if lhs.type != rhs.type:
            raise ValueError(f"lhs.type != rhs.type: {lhs.type} vs. {rhs.type}")
        if isinstance(lhs.type, IntegerType | BoolType):
            return self.create(arith.SubIOp(lhs, rhs)).results[0]
        elif isinstance(lhs.type, FloatType):
            return self.create(arith.SubFOp(lhs, rhs)).results[0]
        else:
            return self.create(tosa.SubOp(lhs, rhs)).results[0]

    def make_mul_op(self, node1: Unary, node2: Unary):
        lhs = self.visit(node1)
        rhs = self.visit(node2)
        if lhs.type != rhs.type:
            raise ValueError(f"lhs.type != rhs.type: {lhs.type} vs. {rhs.type}")
        if isinstance(lhs.type, IntegerType | BoolType):
            return self.create(arith.MulIOp(lhs, rhs)).results[0]
        elif isinstance(lhs.type, FloatType):
            return self.create(arith.MulFOp(lhs, rhs)).results[0]
        else:
            return self.create(tosa.MulOp(lhs, rhs)).results[0]

    def make_div_op(self, node1: Unary, node2: Unary):
        lhs = self.visit(node1)
        rhs = self.visit(node2)
        if lhs.type != rhs.type:
            raise ValueError(f"lhs.type != rhs.type: {lhs.type} vs. {rhs.type}")
        if isinstance(lhs.type, IntegerType | BoolType):
            return self.create(arith.DivSIOp(lhs, rhs)).results[0]
        elif isinstance(lhs.type, FloatType):
            return self.create(arith.DivFOp(lhs, rhs)).results[0]
        else:
            _rhs = self.create(tosa.ReciprocalOp(rhs)).results[0]
            return self.create(tosa.MulOp(lhs, _rhs)).results[0]

    def make_numpy_op(self, op_name, *args, **kwargs):
        if op_name == 'numpy.random.randn':
            return self.create(numpy_dialect.RandomRandnOp(args)).results[0]
        else:
            raise ValueError(f"unsupported op: {op_name}")

    def make_call_op(self, func_op: FuncOp, *args, **kwargs):
        attr = {
            'callee': FlatSymbolRefAttr(StringAttr(func_op.attributes['sym_name'].value))
        }
        return self.create(func.CallOp(args, func_op.function_type.output_types,
                                       attr))

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
    