from ..dialects import func, tensor
from ..dialects.builtin import *
from ..pattern_rewriter import *
from ..pass_manager import *


class ShapeInference(RewritePattern):
    @op_rewrite_pattern
    def match_and_rewrite(self, op: Operation, rewriter: PatternRewriter):
        if hasattr(op, "infer_shapes"):
            op.infer_shapes(op)


class RemoveCastOp(RewritePattern):
    @op_rewrite_pattern
    def match_and_rewrite(self, op: tensor.CastOp, rewriter: PatternRewriter):
        rewriter.replace_all_uses_with(op.results[0], op.operands[0])
        rewriter.erase_op(op)


class ShapeInferencePass(Pass):
    def run_on_operation(self, op: ModuleOp) -> None:
        RewritePatternApplier([ShapeInference(), RemoveCastOp()]).apply(op)
