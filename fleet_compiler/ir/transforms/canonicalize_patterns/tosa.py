from ...dialects import tosa
from ...dialects.builtin import *
from ...pattern_rewriter import *
from ...pass_manager import *


class RemoveCastedIndiceOperandForGather(RewritePattern):
    def match_and_rewrite(self, op: Operation, rewriter: PatternRewriter):
        '''
        %27 = "tosa.cast" (%13) : (tensor<16xi32>) -> tensor<*xf32>
        %22 = "tosa.cast" (%27) : (tensor<*xf32>) -> tensor<*xi32>
        %23 = "tosa.gather" (%20,%22) : (tensor<50257x768xf32>,tensor<*xi32>) -> tensor<*xf32>
        '''
        if isinstance(op.operands[1].type, UnrankedTensorType):
            indice = op.operands[1]
            if isinstance(cast1 := indice.owner(), tosa.CastOp):
                if isinstance(cast1.operands[0].type, UnrankedTensorType):
                    if isinstance(cast2 := cast1.operands[0].owner(), tosa.CastOp):
                        if isinstance(cast2.operands[0].type, RankedTensorType):
                            rewriter.replace_all_uses_with(cast1.results[0], cast2.operands[0])
                            rewriter.erase_op(cast1)
                            rewriter.erase_op(cast2)
                            return True
        return False


class RemoveRedundantCast(RewritePattern):
    def match_and_rewrite(self, op: Operation, rewriter: PatternRewriter):
        '''
        case 1:
        %42 = "tosa.cast" (%16) : (tensor<*xf32>) -> tensor<*xf32>
        '''
        if op.operands[0].type == op.results[0].type and \
            isinstance(op.operands[0].type, UnrankedTensorType):
                rewriter.replace_all_uses_with(op.results[0], op.operands[0])
                rewriter.erase_op(op)
                return True

        '''
        case 2:
        %16 = "tosa.add" (%15, ...) : (...) -> tensor<4xf32>
        %42 = "tosa.cast" (%16) : (tensor<4xf32>) -> tensor<*xf32>
        %43 = "tosa.sub" (%42, ...) : (tensor<*xf32>, ...) -> tensor<*xf32>
        '''
        if isinstance(op.results[0].type, UnrankedTensorType) and \
            isinstance(op.operands[0].type, RankedTensorType):
                rewriter.replace_all_uses_with(op.results[0], op.operands[0])
                rewriter.erase_op(op)
                return True
        
        return False
