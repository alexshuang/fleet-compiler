
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
from functools import reduce
import operator

from typing import Union
from .core import *
from .builder import *
from .dialects.builtin import *
from .dialects.func import *
from .dialects import arith, tosa, numpy as numpy_dialect, func, math, tensor


from fleet_compiler.frontend.lexer import (
    Op as OpCode
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
        with Builder.at_end(block) as builder:
            for o in block.arguments:
                # ast name mapping to ir name
                builder.symbol_table[o.ast_name] = o.name
                # ir name mapping to Value
                builder.symbol_table[o.name] = o

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
        if node.op == OpCode.Assign:
            return self.visit(node.exp2)
        elif node.op in [OpCode.Plus, OpCode.Minus,
                         OpCode.Multiply, OpCode.Divide]:
            return self.make_elementwise_op(node.op, node.exp1, node.exp2)
        elif node.op == OpCode.AT:
            return self.make_matmul_op(node.exp1, node.exp2)
        elif node.op == OpCode.Power:
            return self.make_pow_op(node.exp1, node.exp2)
        # elif node.op in [OpCode.EQ, OpCode.NE, OpCode.GT,
        #             OpCode.GE, OpCode.LT, OpCode.LE]:
        #     return self.make_cmp_op(node.exp1, node.exp2)
        else:
            raise TypeError(f"Interpreter: Unsupport operator {node.op}")

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
        arg_attrs = self.visitSignature(node.signature)
        ast_arg_names, input_types = [], []
        for o in arg_attrs:
            type = FloatType(32) if isinstance(o[1], NoneAttr) else o.type
            input_types.append(UnrankedTensorType(type))
            ast_arg_names.append(o[0])

        entry_block = Block()
        args = [BlockArgument(f'%arg{i}', type, [], entry_block, i, ast_name)
                for i, (type, ast_name) in enumerate(zip(input_types, ast_arg_names))]
        entry_block.arguments = args

        self.visitBlock(node.block, entry_block)

        ret = entry_block.operations[-1]
        assert isinstance(ret, ReturnOp)
        output_types = [o.type for o in ret.operands]

        func_type = FunctionType(input_types, output_types)
        func_op = self.create(func.FuncOp(node.name, func_type, Region([entry_block]),
                                          arg_attrs))
        builder = ImplicitBuilder().get()
        builder.symbol_table[node.name] = func_op

    def visitSignature(self, node: Signature):
        return self.visitParameterList(node.param_list)

    def visitParameterList(self, node: ParameterList):
        return [tuple(self.visitParamentDef(p)) for p in node.params]

    def visitParamentDef(self, node: ParamentDef):
        attr = self.visit(node.init) if node.init else NoneAttr()
        return [node.name, attr]

    def visitReturnStatement(self, node: ReturnStatement):
        ret_val = self.visit(node.ret)
        if not isinstance(ret_val, list):
            ret_val = [ret_val]
        self.create(ReturnOp(ret_val))
        self.has_explicit_ret = True

    def visitBlockEnd(self, node: BlockEnd):
        if not self.has_explicit_ret:
            self.create(ReturnOp([]))
            self.has_explicit_ret = False

    def create(self, op: Operation):
        builder = ImplicitBuilder().get()
        builder.insert(op)
        return op

    def make_elementwise_op(self, opcode: OpCode, node1: Unary, node2: Unary):
        lhs = self.visit(node1)
        rhs = self.visit(node2)

        # boardcast scalar to tensor
        if isinstance(lhs.type, Union[RankedTensorType | UnrankedTensorType]):
            if not isinstance(rhs.type, Union[RankedTensorType | UnrankedTensorType]):
                rhs = self.create(tensor.SplatOp(rhs, lhs.type)).results[0]
        elif isinstance(rhs.type, RankedTensorType | UnrankedTensorType):
            if not isinstance(lhs.type, Union[RankedTensorType | UnrankedTensorType]):
                lhs = self.create(tensor.SplatOp(lhs, rhs.type)).results[0]

        if opcode == OpCode.Plus:
            if isinstance(lhs.type, IntegerType | BoolType):
                return self.create(arith.AddIOp(lhs, rhs)).results[0]
            elif isinstance(lhs.type, FloatType):
                return self.create(arith.AddFOp(lhs, rhs)).results[0]
            else:
                return self.create(tosa.AddOp(lhs, rhs)).results[0]
        elif opcode == OpCode.Minus:
            if isinstance(lhs.type, IntegerType | BoolType):
                return self.create(arith.SubIOp(lhs, rhs)).results[0]
            elif isinstance(lhs.type, FloatType):
                return self.create(arith.SubFOp(lhs, rhs)).results[0]
            else:
                return self.create(tosa.SubOp(lhs, rhs)).results[0]
        elif opcode == OpCode.Multiply:
            if isinstance(lhs.type, IntegerType | BoolType):
                return self.create(arith.MulIOp(lhs, rhs)).results[0]
            elif isinstance(lhs.type, FloatType):
                return self.create(arith.MulFOp(lhs, rhs)).results[0]
            else:
                return self.create(tosa.MulOp(lhs, rhs)).results[0]
        elif opcode == OpCode.Divide:
            if isinstance(lhs.type, IntegerType | BoolType):
                return self.create(arith.DivSIOp(lhs, rhs)).results[0]
            elif isinstance(lhs.type, FloatType):
                return self.create(arith.DivFOp(lhs, rhs)).results[0]
            else:
                _rhs = self.create(tosa.ReciprocalOp(rhs)).results[0]
                return self.create(tosa.MulOp(lhs, _rhs)).results[0]

    def make_numpy_op(self, op_name, *args, **kwargs):
        if op_name == 'numpy.random.randn':
            return self.create(numpy_dialect.Random_RandnOp(args, kwargs)).results[0]
        if op_name == 'numpy.transpose':
            return self.create(numpy_dialect.TransposeOp(args, kwargs)).results[0]
        if op_name == 'numpy.mean':
            return self.create(numpy_dialect.MeanOp(args, kwargs)).results[0]
        if op_name == 'numpy.var':
            return self.create(numpy_dialect.VarOp(args, kwargs)).results[0]
        if op_name == 'numpy.sqrt':
            return self.create(numpy_dialect.SqrtOp(args, kwargs)).results[0]
        else:
            raise ValueError(f"unsupported op: {op_name}")

    def make_call_op(self, func_op: FuncOp, *args, **kwargs):
        attr = {
            'callee': FlatSymbolRefAttr(StringAttr(func_op.attributes['sym_name'].value))
        }
        return self.create(func.CallOp(args, func_op.function_type.output_types,
                                       attr))

    def make_matmul_op(self, node1: Unary, node2: Unary):
        lhs = self.visit(node1)
        rhs = self.visit(node2)
        lhs_type = lhs.type
        rhs_type = rhs.type

        reshape_lhs = False

        if isinstance(lhs_type, RankedTensorType):
            lhs_dims, rhs_dims = lhs_type.dims, rhs_type.dims
            if len(lhs_dims) > len(rhs_dims):
                reshape_lhs = True
                old_outer_dims = lhs_dims[:-1]
                dim0 = reduce(operator.mul, old_outer_dims, 1)
                lhs_dims = [dim0, lhs_dims[-1]]
                new_shape = ArrayAttr(lhs_dims, ArrayType([2], IntegerType(32, True)))
                lhs = self.create(tosa.ReshapeOp(lhs, attrs={'new_shape': new_shape})).results[0]

        out = self.create(tosa.MatmulOp(lhs, rhs)).results[0]

        if reshape_lhs:
            new_out_dims = old_outer_dims + [out.type.dims[-1]]
            new_out_shape = ArrayAttr(new_out_dims, ArrayType([len(old_outer_dims) + 1], IntegerType(32, True)))
            out = self.create(tosa.ReshapeOp(out, attrs={'new_shape': new_out_shape})).results[0]

        return out

    def make_pow_op(self, node1: Unary, node2: Unary):
        lhs = self.visit(node1)
        rhs = self.visit(node2)

        lhs_type, rhs_type = lhs.type, rhs.type
        if isinstance(lhs_type, RankedTensorType):
            if not isinstance(rhs_type, RankedTensorType):
                raise ValueError(f"Invalid lhs/rhs to make power op! {lhs.type} vs. {rhs.type}")
            lhs_type, rhs_type = lhs_type.element_type, rhs_type.element_type

        if isinstance(lhs_type, FloatType):
            if isinstance(rhs_type, IntegerType):
                return self.create(math.FPowIOp(lhs, rhs)).results[0]
            elif isinstance(rhs_type, FloatType):
                return self.create(math.PowFOp(lhs, rhs)).results[0]
        elif isinstance(lhs_type, IntegerType) and isinstance(rhs_type, IntegerType):
            return self.create(math.IPowIOp(lhs, rhs)).results[0]
        else:
            raise ValueError("Invalid lhs/rhs to make power op!")

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
        return ConvertASTtoMLIR(self.module).convert()
