import numpy as np
from fleet_compiler.ir.pattern_rewriter import PatternRewriter
from ..dialects.builtin import *
from ..pattern_rewriter import *
from ..pass_manager import *
from ..dialects import vm, tosa, arith, math, python
from ..dialects.func import FlatSymbolRefAttr


class VmCallOpImpl:
    def convert(self, op: Operation, rewriter: PatternRewriter, sym_name: str = None):
        attrs = op.attributes.copy()
        if not sym_name:
            sym_name = f"device.{op.name.split('.')[-1]}"
        attrs['callee'] = FlatSymbolRefAttr(StringAttr(sym_name))
        new_op = vm.CallOp(op.operands, [o.type for o in op.results], attrs)
        if new_op.results:
            rewriter.replace_op(op, new_op.results)
        else:
            rewriter.insert_op_before(op, new_op)
        rewriter.erase_op(op)
        return True


class PrintOpLowering(RewritePattern, VmCallOpImpl):
    @op_rewrite_pattern
    def match_and_rewrite(self, op: python.PrintOp, rewriter: PatternRewriter) -> bool:
        return self.convert(op, rewriter, "python.print")


class MatmulOpLowering(RewritePattern, VmCallOpImpl):
    @op_rewrite_pattern
    def match_and_rewrite(self, op: tosa.MatmulOp, rewriter: PatternRewriter) -> bool:
        return self.convert(op, rewriter)


class TransposeOpLowering(RewritePattern, VmCallOpImpl):
    @op_rewrite_pattern
    def match_and_rewrite(self, op: tosa.TransposeOp, rewriter: PatternRewriter) -> bool:
        return self.convert(op, rewriter)


class AddOpLowering(RewritePattern, VmCallOpImpl):
    @op_rewrite_pattern
    def match_and_rewrite(self, op: tosa.AddOp, rewriter: PatternRewriter) -> bool:
        return self.convert(op, rewriter)


class SubOpLowering(RewritePattern, VmCallOpImpl):
    @op_rewrite_pattern
    def match_and_rewrite(self, op: tosa.SubOp, rewriter: PatternRewriter) -> bool:
        return self.convert(op, rewriter)


class MulOpLowering(RewritePattern, VmCallOpImpl):
    @op_rewrite_pattern
    def match_and_rewrite(self, op: tosa.MulOp, rewriter: PatternRewriter) -> bool:
        return self.convert(op, rewriter)


class ReciprocalOpLowering(RewritePattern, VmCallOpImpl):
    @op_rewrite_pattern
    def match_and_rewrite(self, op: tosa.ReciprocalOp, rewriter: PatternRewriter) -> bool:
        return self.convert(op, rewriter)


class GatherOpLowering(RewritePattern, VmCallOpImpl):
    @op_rewrite_pattern
    def match_and_rewrite(self, op: tosa.GatherOp, rewriter: PatternRewriter) -> bool:
        return self.convert(op, rewriter)


class PowOpLowering(RewritePattern, VmCallOpImpl):
    @op_rewrite_pattern
    def match_and_rewrite(self, op: tosa.PowOp, rewriter: PatternRewriter) -> bool:
        return self.convert(op, rewriter)


class ReduceSumOpOpLowering(RewritePattern, VmCallOpImpl):
    @op_rewrite_pattern
    def match_and_rewrite(self, op: tosa.ReduceSumOp, rewriter: PatternRewriter) -> bool:
        return self.convert(op, rewriter)


class ReduceMaxOpOpLowering(RewritePattern, VmCallOpImpl):
    @op_rewrite_pattern
    def match_and_rewrite(self, op: tosa.ReduceMaxOp, rewriter: PatternRewriter) -> bool:
        return self.convert(op, rewriter)


class CastOpOpLowering(RewritePattern, VmCallOpImpl):
    @op_rewrite_pattern
    def match_and_rewrite(self, op: tosa.CastOp, rewriter: PatternRewriter) -> bool:
        return self.convert(op, rewriter)


class TanhOpLowering(RewritePattern, VmCallOpImpl):
    @op_rewrite_pattern
    def match_and_rewrite(self, op: tosa.TanhOp, rewriter: PatternRewriter) -> bool:
        return self.convert(op, rewriter)


class ExpOpLowering(RewritePattern, VmCallOpImpl):
    @op_rewrite_pattern
    def match_and_rewrite(self, op: tosa.ExpOp, rewriter: PatternRewriter) -> bool:
        return self.convert(op, rewriter)


class ConstOpLowering(RewritePattern):
    @op_rewrite_pattern
    def match_and_rewrite(self, op: tosa.ConstOp, rewriter: PatternRewriter) -> bool:
        new_op = vm.RodataOp(op.attributes)
        rewriter.replace_op(op, new_op.results)
        rewriter.erase_op(op)
        return True


class SliceOpLowering(RewritePattern, VmCallOpImpl):
    @op_rewrite_pattern
    def match_and_rewrite(self, op: tosa.SliceOp, rewriter: PatternRewriter) -> bool:
        return self.convert(op, rewriter)


class ConcatOpLowering(RewritePattern, VmCallOpImpl):
    @op_rewrite_pattern
    def match_and_rewrite(self, op: tosa.ConcatOp, rewriter: PatternRewriter) -> bool:
        return self.convert(op, rewriter)


class ConvertTosaToVmPass(Pass):
    def run_on_operation(self, op: ModuleOp) -> None:
        RewritePatternApplier([PrintOpLowering(),
                               MatmulOpLowering(),
                               AddOpLowering(),
                               SubOpLowering(),
                               MulOpLowering(),
                               ReciprocalOpLowering(),
                               GatherOpLowering(),
                               PowOpLowering(),
                               ReduceSumOpOpLowering(),
                               ReduceMaxOpOpLowering(),
                               CastOpOpLowering(),
                               ConstOpLowering(),
                               SliceOpLowering(),
                               ConcatOpLowering(),
                               ExpOpLowering(),
                               TanhOpLowering(),
                               TransposeOpLowering()]).apply(op)
