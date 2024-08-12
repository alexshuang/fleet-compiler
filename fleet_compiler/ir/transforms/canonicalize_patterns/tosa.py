from fleet_compiler.ir.pattern_rewriter import PatternRewriter
from ...dialects import tosa, arith
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
                        if isinstance(cast2.operands[0].type, RankedTensorType) and \
                            isinstance(cast2.operands[0].type.element_type, IntegerType):
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
            not isinstance(op.operands[0].type, UnrankedTensorType):
                rewriter.replace_all_uses_with(op.results[0], op.operands[0])
                rewriter.erase_op(op)
                return True
        
        return False


class CastIndiceIntergeToTensor(RewritePattern):
    def match_and_rewrite(self, op: Operation, rewriter: PatternRewriter) -> bool:
        '''
        %272 = "arith.constant" () {value = 2: i32} : () -> i32
        %274 = "tosa.gather" (%261,%272) : (tensor<*xf32>,i32) -> tensor<*xf32>
        '''
        indice = op.operands[1]
        if isinstance(indice.type, IntegerType) and isinstance(indice.owner(), arith.ConstantOp):
            new_value = [indice.owner().attributes['value'].value]
            attr = DenseIntOrFPElementsAttr(new_value, RankedTensorType([1], IntegerType(32, True)))
            const = tosa.ConstOp({'value': attr})
            rewriter.replace_op(indice.owner(), const.results)
            rewriter.erase_op(indice.owner())
            return True
        return False


class RemoveRedundantReshape(RewritePattern):
    def match_and_rewrite(self, op: Operation, rewriter: PatternRewriter):
        '''
        %538 = "tosa.reshape" (%537) {new_shape = [16, 1]} : (tensor<16x1xf32>) -> tensor<16x1xf32>
        '''
        if op.operands[0].type == op.results[0].type:
            rewriter.replace_all_uses_with(op.results[0], op.operands[0])
            rewriter.erase_op(op)
            return True
        return False
