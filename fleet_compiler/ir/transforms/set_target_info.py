from fleet_compiler.ir.pattern_rewriter import PatternRewriter
from ..core import *
from ..dialects.builtin import *
from ..pattern_rewriter import *
from ..pass_manager import *


class SetTargetInfo(RewritePattern):
    def __init__(self, target_info):
        self.target_info = target_info

    @op_rewrite_pattern
    def match_and_rewrite(self, op: ModuleOp, rewriter: PatternRewriter) -> bool:
        op.attributes['target_info'] = self.target_info


class SetTargetInfoPass(Pass):
    def __init__(self, target_backend: str):
        super().__init__()
        self.target_info = DictAttr({
            'target_backend': StringAttr(target_backend)
        })

    def run_on_operation(self, op: ModuleOp) -> None:
        RewritePatternApplier([SetTargetInfo(self.target_info)]).apply(op)
