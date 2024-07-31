from ..dialects import func, tensor
from ..dialects.builtin import *
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
        assert isinstance((module := op.parent_node.parent_node.parent_node), ModuleOp), \
            "Nested functions are not supported"
        assert isinstance(block := op.parent_node, Block)

        func_name = op.attributes['callee'].sym_name.value
        func_op = self._get_func_op_by_name(module, func_name)
        new_func_op = func_op.clone()
        inlined_block = new_func_op.regions[0].blocks[0]

        # cast ranked tensors to unranked tensors
        cast_ops = [tensor.CastOp(operand, type)
                    for operand, type in zip(op.operands,
                                             new_func_op.attributes['function_type'].input_types)]
        for o in cast_ops:
            block.insert_before(o, op)
        
        # replace operands
        args = inlined_block.arguments
        new_operands = [o.results[0] for o in cast_ops]
        for old, new in zip(args, new_operands):
            rewriter.replace_all_uses_with(old, new)
        
        # inline block before call op
        new_ops = [o for o in inlined_block.operations if not isinstance(o, func.ReturnOp)]
        for o in new_ops:
            o.parent = None
            block.insert_before(o, op)

        last_op = new_ops[-1]

        # replace results
        # MUST: return value are the previous output
        for old, new in zip(op.results, last_op.results):
            rewriter.replace_all_uses_with(old, new)
            for use in old.uses:
                new.uses.append(Use(use.operation, use.index))

        # erase call op & func decl op
        rewriter.erase_op(op)
        rewriter.erase_op(func_op)


class InlineFunctionPass(Pass):
    def run_on_operation(self, op: ModuleOp) -> None:
        RewritePatternApplier([InlineFunction()]).apply(op)
