import numpy as np
from fleet_compiler.ir.pattern_rewriter import PatternRewriter
from ..dialects.builtin import *
from ..pattern_rewriter import *
from ..pass_manager import *
from ..dialects.numpy import *
from ..dialects import math, tosa


class SqrtOpLowering(RewritePattern):
    @op_rewrite_pattern
    def match_and_rewrite(self, op: SqrtOp, rewriter: PatternRewriter) -> bool:
        new_op = math.SqrtOp(op.operands[0])
        rewriter.replace_op(op, new_op.results[0])
        rewriter.erase_op(op)
        return True


class MeanOpLowering(RewritePattern):
    @op_rewrite_pattern
    def match_and_rewrite(self, op: MeanOp, rewriter: PatternRewriter) -> bool:
        input_dims = op.operands[0].type.dims
        assert isinstance((axis_attr := op.attributes['axis']), ArrayAttr) and \
                axis_attr.type.dims == 1, "Not support multi axis mean."

        # reduce_sum_attrs = {'axis': IntegerAttr(axis_attr.value[0], IntegerType(64, True))}
        reduce_sum_attrs = op.attributes
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
    def match_and_rewrite(self, op: VarOp, rewriter: PatternRewriter) -> bool:
        input_dims = op.operands[0].type.dims
        assert isinstance((axis_attr := op.attributes['axis']), ArrayAttr) and \
                axis_attr.type.dims == 1, "Not support multi axis mean."

        # reduce_sum_attrs = {'axis': IntegerAttr(axis_attr.value[0], IntegerType(64, True))}
        reduce_sum_attrs = op.attributes
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

        reduced2 = tosa.ReduceSumOp(square.results[0], reduce_sum_attrs)
        rewriter.insert_op_before(op, reduced2)

        div2 = tosa.MulOp(reduced2.results[0], rec.results[0])
        rewriter.insert_op_before(op, div2)

        out = tosa.ReshapeOp(div2.results[0], {'new_shape': new_shape})
        rewriter.insert_op_before(op, out)

        for old, new in zip(op.results, out.results):
            rewriter.replace_all_uses_with(old, new)
        rewriter.erase_op(op)
        return True


class RandomSeedOpLowering(RewritePattern):
    @op_rewrite_pattern
    def match_and_rewrite(self, op: Random_SeedOp, rewriter: PatternRewriter) -> bool:
        seed = op.operands[0].owner().attributes['value'].value
        np.random.seed(seed)
        rewriter.replace_all_uses_with(op.results[0], op.operands[0])
        rewriter.erase_op(op)
        return True


class RandomRandnOpLowering(RewritePattern):
    @op_rewrite_pattern
    def match_and_rewrite(self, op: Random_RandnOp, rewriter: PatternRewriter) -> bool:
        shape = [o.owner().attributes['value'].value for o in op.operands]
        init = np.random.randn(*shape)
        const = tosa.ConstOp({'value': DenseIntOrFPElementsAttr(init.reshape(-1),
                                        RankedTensorType(shape, FloatType(32)))})
        rewriter.replace_op(op, const.results)
        rewriter.erase_op(op)
        return True


class TransposeOpLowering(RewritePattern):
    @op_rewrite_pattern
    def match_and_rewrite(self, op: TransposeOp, rewriter: PatternRewriter) -> bool:
        axes = op.attributes['axes'].value
        perms = tosa.ConstOp({'value': DenseIntOrFPElementsAttr(axes,
                                            RankedTensorType([len(axes)], IntegerType(32, True)))})
        rewriter.insert_op_before(op, perms)
        tosa_trans = tosa.TransposeOp(op.operands[0], perms.results[0])
        rewriter.replace_op(op, tosa_trans.results)
        rewriter.erase_op(op)
        return True


class TriOpLowering(RewritePattern):
    @op_rewrite_pattern
    def match_and_rewrite(self, op: TriOp, rewriter: PatternRewriter) -> bool:
        const_value = op.operands[0].owner().attributes['value'].value
        value = np.tri(const_value)
        init = tosa.ConstOp({'value': DenseIntOrFPElementsAttr(value.reshape(-1),
                                            RankedTensorType(value.shape, IntegerType(32, True)))})
        rewriter.replace_op(op, init.results)
        rewriter.erase_op(op)
        return True


class TanhOpLowering(RewritePattern):
    @op_rewrite_pattern
    def match_and_rewrite(self, op: TanhOp, rewriter: PatternRewriter) -> bool:
        new_op = tosa.TanhOp(op.operands[0])
        rewriter.replace_op(op, new_op.results)
        rewriter.erase_op(op)
        return True


class ExpOpLowering(RewritePattern):
    @op_rewrite_pattern
    def match_and_rewrite(self, op: ExpOp, rewriter: PatternRewriter) -> bool:
        new_op = tosa.ExpOp(op.operands[0])
        rewriter.replace_op(op, new_op.results)
        rewriter.erase_op(op)
        return True


class MaxOpLowering(RewritePattern):
    @op_rewrite_pattern
    def match_and_rewrite(self, op: MaxOp, rewriter: PatternRewriter) -> bool:
        # new_op = tosa.ReduceMaxOp(op.operands[0], {'axis': op.attributes['axis']})
        new_op = tosa.ReduceMaxOp(op.operands[0], op.attributes)
        rewriter.replace_op(op, new_op.results)
        rewriter.erase_op(op)
        return True


class SumOpLowering(RewritePattern):
    @op_rewrite_pattern
    def match_and_rewrite(self, op: SumOp, rewriter: PatternRewriter) -> bool:
        # new_op = tosa.ReduceSumOp(op.operands[0], {'axis': op.attributes['axis']})
        new_op = tosa.ReduceSumOp(op.operands[0], op.attributes)
        rewriter.replace_op(op, new_op.results)
        rewriter.erase_op(op)
        return True


class HstackOpLowering(RewritePattern):
    @op_rewrite_pattern
    def match_and_rewrite(self, op: HstackOp, rewriter: PatternRewriter) -> bool:
        new_op = tosa.ConcatOp(op.operands, None, {'axis': IntegerAttr(-1, IntegerType(32, True))})
        rewriter.replace_op(op, new_op.results)
        rewriter.erase_op(op)
        return True


class SplitOpLowering(RewritePattern):
    @op_rewrite_pattern
    def match_and_rewrite(self, op: SplitOp, rewriter: PatternRewriter) -> bool:
        input = op.operands[0]
        n_split = op.operands[1].owner().attributes['value'].value
        # assert all operands have the same shape
        dims = op.operands[0].type.dims

        axis = [len(dims) + o if o < 0 else o for o in op.attributes['axis'].value]
        assert len(axis) == 1
        axis = axis[0]
        dim = dims[axis]
        assert dim % n_split == 0

        tile_dim = dim // n_split
        tile_size = [o if i != axis else tile_dim for i, o in enumerate(dims)]
        size_attr = ArrayAttr(tile_size, ArrayType(len(tile_size), IntegerType(32, True)))

        slices = []
        for i in range(n_split):
            start = [0 if j != axis else i * tile_dim for j in range(len(dims))]
            start_attr = ArrayAttr(start, ArrayType(len(start), IntegerType(32, True)))
            slices.append(tosa.SliceOp(input, {'start': start_attr, 'size': size_attr}))
        rewriter.insert_op_before(op, slices)

        reshaped = []
        for o in slices:
            res = o.results[0]
            out_dims = [1] + res.type.dims
            new_shape = ArrayAttr(out_dims, ArrayType([len(out_dims)], IntegerType(32, True)))
            reshaped.append(tosa.ReshapeOp(res, {'new_shape': new_shape}))
        rewriter.insert_op_before(op, reshaped)

        new_operands = [o.results[0] for o in reshaped]
        output_shape = RankedTensorType([n_split] + tile_size, input.type.element_type)
        new_op = tosa.ConcatOp(new_operands, output_shape,
                               {'axis': IntegerAttr(0, IntegerType(32, True))})
        rewriter.replace_op(op, new_op.results)
        rewriter.erase_op(op)
        return True


class LowerNumpyPass(Pass):
    def run_on_operation(self, op: ModuleOp) -> None:
        RewritePatternApplier([SqrtOpLowering(),
                               MeanOpLowering(),
                               RandomSeedOpLowering(),
                               RandomRandnOpLowering(),
                               TransposeOpLowering(),
                               TriOpLowering(),
                               TanhOpLowering(),
                               ExpOpLowering(),
                               MaxOpLowering(),
                               SumOpLowering(),
                               HstackOpLowering(),
                               SplitOpLowering(),
                               VarOpLowering()]).apply(op)
