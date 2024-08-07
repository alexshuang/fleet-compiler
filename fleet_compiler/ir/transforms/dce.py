from ..dialects import func, tensor
from ..dialects.builtin import *
from ..pattern_rewriter import *
from ..pass_manager import *
from ..traits import *


class DCE(RewritePattern):
    def match_and_rewrite(self, op: Operation, rewriter: PatternRewriter):
        if Pure() in op.traits:
            for o in op.results:
                if len(o.uses) > 0:
                    return False
            rewriter.erase_op(op)
            return True
        return False


class DeadCodeEliminationPass(Pass):
    def run_on_operation(self, op: ModuleOp) -> None:
        RewritePatternApplier([DCE()], reverse_ops=True).apply(op)
