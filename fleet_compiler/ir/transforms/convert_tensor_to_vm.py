import numpy as np
from fleet_compiler.ir.pattern_rewriter import PatternRewriter
from ..dialects.builtin import *
from ..pattern_rewriter import *
from ..pass_manager import *
from ..dialects import vm, tensor
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


class SplatOpLowering(RewritePattern, VmCallOpImpl):
    @op_rewrite_pattern
    def match_and_rewrite(self, op: tensor.SplatOp, rewriter: PatternRewriter) -> bool:
        return self.convert(op, rewriter)


class ConvertTensorToVmPass(Pass):
    def run_on_operation(self, op: ModuleOp) -> None:
        RewritePatternApplier([SplatOpLowering()]).apply(op)
