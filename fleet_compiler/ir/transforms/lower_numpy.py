from fleet_compiler.ir.pattern_rewriter import PatternRewriter
from ..dialects.builtin import *
from ..pattern_rewriter import *
from ..pass_manager import *
from ..dialects import numpy
from ..dialects import math, tosa


class SqrtOpLowering(RewritePattern):
    @op_rewrite_pattern
    def match_and_rewrite(self, op: numpy.SqrtOp, rewriter: PatternRewriter) -> bool:
        new_op = math.SqrtOp(op.operands[0])
        rewriter.replace_op(op, new_op.results[0])
        rewriter.erase_op(op)
        return True


class MeanOpLowering(RewritePattern):
    @op_rewrite_pattern
    def match_and_rewrite(self, op: numpy.MeanOp, rewriter: PatternRewriter) -> bool:
        input_dims = op.operands[0].type.dims
        assert isinstance((axis_attr := op.attributes['axis']), ArrayAttr) and \
                axis_attr.type.dims == 1, "Not support multi axis mean."

        reduce_sum_attrs = {'axis': IntegerAttr(axis_attr.value[0], IntegerType(64, True))}
        reduced = tosa.ReduceSumOp(op.operands[0], reduce_sum_attrs)
        rewriter.insert_op_before(op, reduced)

        axis = op.attributes['axis'].value[0]
        dim = input_dims[axis]
        attr = DenseIntOrFPElementsAttr(dim, RankedTensorType([1], IntegerType(32, True)))
        c_dim = tosa.ConstOp({'value': attr})
        rewriter.insert_op_before(op, c_dim)

        cast = tosa.CastOp(c_dim.results[0], reduced.results[0].type)
        rewriter.insert_op_before(op, cast)

        rec = tosa.ReciprocalOp(cast.results[0])
        div = tosa.MulOp(reduced.results[0], rec.results[0])
        rewriter.insert_op_before(op, [rec, div])

        out_dims = input_dims.copy()
        out_dims[axis] = 1
        new_shape = ArrayAttr(out_dims, ArrayType([len(out_dims)], IntegerType(32, True)))
        reshaped = tosa.ReshapeOp(div.results[0], {'new_shape': new_shape})
        rewriter.insert_op_before(op, reshaped)

        for old, new in zip(op.results, reshaped.results):
            rewriter.replace_all_uses_with(old, new)
        rewriter.erase_op(op)
        return True


class VarOpLowering(RewritePattern):
    @op_rewrite_pattern
    def match_and_rewrite(self, op: numpy.VarOp, rewriter: PatternRewriter) -> bool:
        input_dims = op.operands[0].type.dims
        assert isinstance((axis_attr := op.attributes['axis']), ArrayAttr) and \
                axis_attr.type.dims == 1, "Not support multi axis mean."

        reduce_sum_attrs = {'axis': IntegerAttr(axis_attr.value[0], IntegerType(64, True))}
        reduced = tosa.ReduceSumOp(op.operands[0], reduce_sum_attrs)
        rewriter.insert_op_before(op, reduced)

        axis = op.attributes['axis'].value[0]
        dim = input_dims[axis]
        attr = DenseIntOrFPElementsAttr(dim, RankedTensorType([1], IntegerType(32, True)))
        c_dim = tosa.ConstOp({'value': attr})
        rewriter.insert_op_before(op, c_dim)

        cast = tosa.CastOp(c_dim.results[0], reduced.results[0].type)
        rewriter.insert_op_before(op, cast)

        rec = tosa.ReciprocalOp(cast.results[0])
        div = tosa.MulOp(reduced.results[0], rec.results[0])
        rewriter.insert_op_before(op, [rec, div])

        out_dims = input_dims.copy()
        out_dims[axis] = 1
        new_shape = ArrayAttr(out_dims, ArrayType([len(out_dims)], IntegerType(32, True)))
        reshaped = tosa.ReshapeOp(div.results[0], {'new_shape': new_shape})
        rewriter.insert_op_before(op, reshaped)

        cast2 = tosa.CastOp(reshaped.results[0], op.operands[0].type)
        sub = tosa.SubOp(op.operands[0], cast2.results[0])
        rewriter.insert_op_before(op, [cast2, sub])

        square = tosa.MulOp(sub.results[0], sub.results[0])
        rewriter.insert_op_before(op, square)

        reduced2 = tosa.ReduceSumOp(op.operands[0], reduce_sum_attrs)
        rewriter.insert_op_before(op, reduced2)

        div2 = tosa.MulOp(reduced2.results[0], rec.results[0])
        rewriter.insert_op_before(op, div2)

        out = tosa.ReshapeOp(div2.results[0], {'new_shape': new_shape})
        rewriter.insert_op_before(op, out)

        for old, new in zip(op.results, out.results):
            rewriter.replace_all_uses_with(old, new)
        rewriter.erase_op(op)
        return True


class LowerNumpyPass(Pass):
    def run_on_operation(self, op: ModuleOp) -> None:
        RewritePatternApplier([SqrtOpLowering(),
                               MeanOpLowering(),
                               VarOpLowering()]).apply(op)
