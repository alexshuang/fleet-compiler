from fleet_compiler.ir.pattern_rewriter import PatternRewriter
from ...dialects import vm
from ...dialects.builtin import *
from ...dialects.func import FlatSymbolRefAttr
from ...pattern_rewriter import *
from ...pass_manager import *


class ReplaceReciprocalMulWithDiv(RewritePattern):
    def match_and_rewrite(self, op: Operation, rewriter: PatternRewriter):
        '''
        %2891 = "vm.call" @device.reciprocal(%2975) : (tensor<16x1xf32>) -> (tensor<16x1xf32>)
        %2855 = "vm.call" @device.mul(%2954,%2891) : (tensor<16x1xf32>,tensor<16x1xf32>) -> (tensor<16x1xf32>)
        '''
        res = False
        all_uses_are_mul = True
        if type(op) is vm.CallOp and op.attributes['callee'].sym_name.value == "device.reciprocal":
            val = op.results[0]
            attrs = { 'callee': FlatSymbolRefAttr(StringAttr("device.div")) }
            removed_ops = []
            for use in val.uses:
                if type(mul_op := use.operation) is vm.CallOp and mul_op.attributes['callee'].sym_name.value == "device.mul":
                    dividend = op.operands[0]
                    if use.index == 0:
                        new_op = vm.CallOp([mul_op.operands[1], dividend],
                                           [mul_op.results[0].type], attrs)
                    else:
                        new_op = vm.CallOp([mul_op.operands[0], dividend],
                                           [mul_op.results[0].type], attrs)
                    rewriter.replace_op(mul_op, new_op.results)
                    removed_ops.append(mul_op)
                    res = True
                else:
                    all_uses_are_mul = False
            
            for o in removed_ops:
                rewriter.erase_op(o)
            
            if all_uses_are_mul:
                rewriter.erase_op(op)
        return res
