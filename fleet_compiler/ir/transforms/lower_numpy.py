from fleet_compiler.ir.pattern_rewriter import PatternRewriter
from ..dialects.builtin import *
from ..pattern_rewriter import *
from ..pass_manager import *
from ..dialects import numpy
from ..dialects import math


class SqrtOpLowering(RewritePattern):
    @op_rewrite_pattern
    def match_and_rewrite(self, op: numpy.SqrtOp, rewriter: PatternRewriter) -> bool:
        new_op = math.SqrtOp(op.operands[0])
        rewriter.replace_op(op, new_op.results[0])
        rewriter.erase_op(op)
        return True


class LowerNumpyPass(Pass):
    def run_on_operation(self, op: ModuleOp) -> None:
        RewritePatternApplier([SqrtOpLowering()]).apply(op)
