import numpy as np
from fleet_compiler.ir.pattern_rewriter import PatternRewriter
from ..dialects.builtin import *
from ..pattern_rewriter import *
from ..pass_manager import *
from ..dialects import vm, arith
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


class ConstantOpLowering(RewritePattern):
    @op_rewrite_pattern
    def match_and_rewrite(self, op: arith.ConstantOp, rewriter: PatternRewriter) -> bool:
        attr = op.attributes['value']
        if isinstance(attr, BoolAttr):
            val = attr.value
            attr = {'value': IntegerAttr(val, IntegerType(32, True))}
            new_op = vm.Const_I32Op(attr) if val else vm.Const_I32_ZeroOp(attr)
        elif isinstance(attr, IntegerAttr):
            new_op = vm.Const_I32Op(op.attributes) if attr.value else vm.Const_I32_ZeroOp(op.attributes)
        elif isinstance(attr, FloatAttr):
            new_op = vm.Const_F32Op(op.attributes) if attr.value else vm.Const_F32_ZeroOp(op.attributes)
        else:
            new_op = vm.ConstOp(op.attributes)
        rewriter.replace_op(op, new_op.results)
        rewriter.erase_op(op)
        return True


class DivSIOpLowering(RewritePattern, VmCallOpImpl):
    @op_rewrite_pattern
    def match_and_rewrite(self, op: arith.DivSIOp, rewriter: PatternRewriter) -> bool:
        return self.convert(op, rewriter, "device.div")


class SubIOpLowering(RewritePattern, VmCallOpImpl):
    @op_rewrite_pattern
    def match_and_rewrite(self, op: arith.SubIOp, rewriter: PatternRewriter) -> bool:
        return self.convert(op, rewriter, "device.sub")


class MulIOpLowering(RewritePattern, VmCallOpImpl):
    @op_rewrite_pattern
    def match_and_rewrite(self, op: arith.MulIOp, rewriter: PatternRewriter) -> bool:
        return self.convert(op, rewriter, "device.mul")


class ConvertArithToVmPass(Pass):
    def run_on_operation(self, op: ModuleOp) -> None:
        RewritePatternApplier([ConstantOpLowering(),
                               SubIOpLowering(),
                               MulIOpLowering(),
                               DivSIOpLowering()]).apply(op)
