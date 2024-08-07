from ..dialects import func, tensor
from ..dialects.builtin import *
from ..pattern_rewriter import *
from ..pass_manager import *


class Canonicalize(RewritePattern):
    def match_and_rewrite(self, op: Operation, rewriter: PatternRewriter):
        if not op.hasCanonicalizer:
            return False

        patterns = op.get_canonicalize_patterns()
        for p in patterns:
            if p.match_and_rewrite(op, rewriter):
                return True
        return False


class CanonicalizePass(Pass):
    def run_on_operation(self, op: ModuleOp) -> None:
        RewritePatternApplier([Canonicalize()]).apply(op)
