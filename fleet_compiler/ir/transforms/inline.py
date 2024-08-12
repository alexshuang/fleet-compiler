from fleet_compiler.ir.pattern_rewriter import PatternRewriter
from ..dialects import func, tosa
from ..dialects.builtin import *
from ..dialects.func import *
from ..pattern_rewriter import *
from ..pass_manager import *


class InlineFunction(RewritePattern):
    func_op_table: dict[str, func.FuncOp] | None = None

    def _get_func_op_by_name(self, module: ModuleOp, name: str):
        if self.func_op_table is None:
            self.func_op_table = {o.attributes['sym_name'].value:o
                                  for o in module.operations if isinstance(o, func.FuncOp)}
        return self.func_op_table[name]

    @op_rewrite_pattern
    def match_and_rewrite(self, op: func.CallOp, rewriter: PatternRewriter):
        module = op.parent_node.parent_node if isinstance(op.parent_node, FuncOp) \
            else op.parent_node
        func_name = op.attributes['callee'].sym_name.value
        func_op = self._get_func_op_by_name(module, func_name)
        new_func_op = func_op.clone()
        inlined_block = new_func_op.regions[0].blocks[0]

        # cast ranked tensors to unranked tensors
        cast_ops = [tosa.CastOp(operand, type)
                    for operand, type in zip(op.operands,
                                             new_func_op.attributes['function_type'].input_types)]
        rewriter.insert_op_before(op, cast_ops)
        op.operands = [o.results[0] for o in cast_ops]

        # replace operands
        args = inlined_block.arguments
        for old, new in zip(args, op.operands):
            rewriter.replace_all_uses_with(old, new)
        
        # inline block before call op
        ret_values = []
        for o in inlined_block.operations:
            if not isinstance(o, func.ReturnOp):
                rewriter.insert_op_before(op, [o])
            else:
                ret_values = o.operands

        assert len(ret_values) > 0

        # replace results
        for old, new in zip(op.results, ret_values):
            rewriter.replace_all_uses_with(old, new)

        # erase call op & func decl op
        rewriter.erase_op(op)
        # set flag to remove the func op by RemoveUnusedFunction 
        func_op.attributes['visibility'] = StringAttr("private")


class RemoveUnusedFunction(RewritePattern):
    used_funcs: set[str] | None = None

    def should_remove(self, op: func.FuncOp):
        if op.attributes['visibility'] != StringAttr("private"):
            return False

        if not self.used_funcs:
            assert isinstance(module := op.parent_node, ModuleOp)
            self.used_funcs = {o.attributes['callee'].sym_name.value
                               for o in module.operations
                               if isinstance(o, CallOp)}
        return op.attributes['sym_name'].value not in self.used_funcs

    @op_rewrite_pattern
    def match_and_rewrite(self, op: func.FuncOp, rewriter: PatternRewriter) -> bool:
        if self.should_remove(op):
            rewriter.erase_op(op)


class SetAllPrivateFunctions(RewritePattern):
    @op_rewrite_pattern
    def match_and_rewrite(self, op: func.FuncOp, rewriter: PatternRewriter) -> bool:
        op.attributes['visibility'] = StringAttr("private")


class InlineFunctionPass(Pass):
    def run_on_operation(self, op: ModuleOp) -> None:
        RewritePatternApplier([InlineFunction(),
                               SetAllPrivateFunctions(),
                               RemoveUnusedFunction()]).apply(op)
