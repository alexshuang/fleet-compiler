# ===- syntax.py -------------------------------------------------------------
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# ===---------------------------------------------------------------------------

from abc import ABC, abstractmethod

from .lexer import *
from .symbolic import FunctionSymbol, VariableSymbol


class AstNode(ABC):
    @abstractmethod
    def accept(self, visitor):
        pass


class Expression(AstNode):
    def __init__(self) -> None:
        super().__init__()


class Statement(AstNode):
    def __init__(self) -> None:
        super().__init__()


class ParamentDef(AstNode):
    def __init__(self, name: str, type=None, init: Expression = None) -> None:
        super().__init__()
        self.name = name
        self.type = type
        self.init = init

    def accept(self, visitor):
        return visitor.visitParamentDef(self)


class ParameterList(AstNode):
    def __init__(self, params: list) -> None:
        super().__init__()
        self.params = params

    def accept(self, visitor):
        return visitor.visitParameterList(self)


class Signature(AstNode):
    def __init__(self, param_list: ParameterList) -> None:
        super().__init__()
        self.param_list = param_list

    def accept(self, visitor):
        return visitor.visitSignature(self)
    

class ImportStatement(Statement):
    def __init__(self, package: str, alias: str = "") -> None:
        super().__init__()
        self.package = package
        self.alias = alias
    
    def accept(self, visitor):
        return visitor.visitImportStatement(self)


class ExpressionStatement(Statement):
    def __init__(self, exp: Expression) -> None:
        super().__init__()
        self.exp = exp
    
    def accept(self, visitor):
        return visitor.visitExpressionStatement(self)


class Block(Statement):
    def __init__(self, stmts: list) -> None:
        super().__init__()
        self.stmts = stmts
        self.indent = None

    def accept(self, visitor):
        return visitor.visitBlock(self)


class BlockEnd(Statement):
    def __init__(self) -> None:
        super().__init__()

    def accept(self, visitor):
        return visitor.visitBlockEnd(self)


class Branch(Statement):
    def __init__(self, cond: Expression, block: Block):
        self.is_else_branch = True if cond is None else False
        self.cond = cond
        self.block = block
    
    def accept(self, visitor):
        return visitor.visitBranch(self)


class IfStatement(Statement):
    def __init__(self, branches: list) -> None:
        super().__init__()
        self.branches = branches
    
    def accept(self, visitor):
        return visitor.visitIfStatement(self)


class AstModule(AstNode):
    def __init__(self, block: Block) -> None:
        self.block = block

    def accept(self, visitor):
        return visitor.visitModule(self)


class FunctionDef(Statement):
    def __init__(self, name: str, signautre: Signature, block: Block) -> None:
        super().__init__()
        self.name = name
        self.signature = signautre
        self.block = block
    
    def accept(self, visitor):
        return visitor.visitFunctionDef(self)


class ArgumentList(AstNode):
    def __init__(self, args: list) -> None:
        super().__init__()
        self.args = args

    def accept(self, visitor):
        return visitor.visitArgumentList(self)


class Argument(AstNode):
    def __init__(self) -> None:
        super().__init__()


class PositionalArgument(Argument):
    def __init__(self, index, value: Expression) -> None:
        super().__init__()
        self.index = index
        self.value = value

    def accept(self, visitor):
        return visitor.visitPositionalArgument(self)


class KeywordArgument(Argument):
    def __init__(self, name: str, value: Expression) -> None:
        super().__init__()
        self.name = name
        self.value = value

    def accept(self, visitor):
        return visitor.visitKeywordArgument(self)


class FunctionCall(Expression):
    def __init__(self, name: str, arg_list: ArgumentList, sym: FunctionSymbol = None) -> None:
        super().__init__()
        self.name = name
        self.arg_list = arg_list
        self.sym = sym
    
    def accept(self, visitor):
        return visitor.visitFunctionCall(self)


class VariableDef(Statement):
    def __init__(self, name: str, type, init) -> None:
        super().__init__()
        self.name = name
        self.type = type
        self.init = init
    
    def accept(self, visitor):
        return visitor.visitVariableDef(self)


class Variable(Expression):
    def __init__(self, name: str, sym: VariableSymbol = None) -> None:
        super().__init__()
        self.name = name
        self.sym = sym
    
    def accept(self, visitor):
        return visitor.visitVariable(self)


class StringLiteral(Expression):
    def __init__(self, data: str) -> None:
        super().__init__()
        self.data = data
    
    def accept(self, visitor):
        return visitor.visitStringLiteral(self)


class IntegerLiteral(Expression):
    def __init__(self, value: int) -> None:
        super().__init__()
        self.value = value
    
    def accept(self, visitor):
        return visitor.visitIntegerLiteral(self)


class DecimalLiteral(Expression):
    def __init__(self, value: float) -> None:
        super().__init__()
        self.value = value
    
    def accept(self, visitor):
        return visitor.visitDecimalLiteral(self)


class BooleanLiteral(Expression):
    def __init__(self, value: bool) -> None:
        super().__init__()
        self.value = value
    
    def accept(self, visitor):
        return visitor.visitBooleanLiteral(self)


class NoneLiteral(Expression):
    def __init__(self) -> None:
        super().__init__()
        self.value = None
    
    def accept(self, visitor):
        return visitor.visitNoneLiteral(self)


class Unary(Expression):
    def __init__(self, exp: Expression) -> None:
        super().__init__()
        self.exp = exp
    
    def accept(self, visitor):
        return visitor.visitUnary(self)


class Binary(Expression):
    def __init__(self, op: Op, exp1: Unary, exp2: Unary) -> None:
        super().__init__()
        self.op = op
        self.exp1 = exp1
        self.exp2 = exp2
    
    def accept(self, visitor):
        return visitor.visitBinary(self)


class ReturnStatement(Statement):
    def __init__(self, ret: Expression = None) -> None:
        super().__init__()
        self.ret = ret
    
    def accept(self, visitor):
        return visitor.visitReturnStatement(self)


class EmptyStatement(Statement):
    def __init__(self) -> None:
        super().__init__()
    
    def accept(self, visitor):
        return visitor.visitEmptyStatement(self)


class Slice(Expression):
    def __init__(self, exps: list, omitted_first_dim: bool = False, omitted_last_dim: bool = False) -> None:
        super().__init__()
        self.exps = exps
        self.omitted_first_dim = omitted_first_dim
        self.omitted_last_dim = omitted_last_dim
    
    def accept(self, visitor):
        return visitor.visitSlice(self)
    

class SliceStatement(Statement):
    def __init__(self, name, slice_obj: Slice) -> None:
        super().__init__()
        self.name = name
        self.slice = slice_obj
        self.sym = None
    
    def accept(self, visitor):
        return visitor.visitSliceStatement(self)


class AstVisitor:
    def visit(self, node: AstNode):
        return node.accept(self)

    def visitModule(self, node: AstModule):
        return self.visitBlock(node.block)

    def visitBlock(self, node: Block):
        for o in node.stmts:
            self.visit(o)

    def visitBlockEnd(self, node: BlockEnd):
        pass

    def visitFunctionDef(self, node: FunctionDef):
        self.visitSignature(node.signature)
        self.visitBlock(node.block)
    
    def visitSignature(self, node: Signature):
        self.visitParameterList(node.param_list)

    def visitParameterList(self, node: ParameterList):
        for p in node.params:
            self.visitParamentDef(p)
    
    def visitParamentDef(self, node: ParamentDef):
        if node.init:
            self.visit(node.init)

    def visitFunctionCall(self, node: FunctionCall):
        if node.arg_list:
            self.visitArgumentList(node.arg_list) 

    def visitArgumentList(self, node: ArgumentList):
        for o in node.args:
            self.visit(o)
        
    def visitPositionalArgument(self, node: PositionalArgument):
        if node.value:
            return self.visit(node.value)

    def visitKeywordArgument(self, node: KeywordArgument):
        if node.value:
            return {node.name, self.visit(node.value)}

    def visitReturnStatement(self, node: ReturnStatement):
        if node.ret:
            self.visit(node.ret)

    def visitExpressionStatement(self, node: ExpressionStatement):
        return self.visit(node.exp)

    def visitVariableDef(self, node: VariableDef):
        if node.init:
            self.visit(node.init)
    
    def visitVariable(self, node: Variable):
        pass

    def visitStringLiteral(self, node: StringLiteral):
        return '"' + node.data + '"'

    def visitIntegerLiteral(self, node: IntegerLiteral):
        return node.value

    def visitDecimalLiteral(self, node: DecimalLiteral):
        return node.value

    def visitBooleanLiteral(self, node: BooleanLiteral):
        return node.value

    def visitNoneLiteral(self, node: NoneLiteral):
        return None

    def visitEmptyStatement(self, node: EmptyStatement):
        pass

    def visitImportStatement(self, node: ImportStatement):
        pass
    
    def visitUnary(self, node: Unary):
        return self.visit(node.exp)
    
    def visitBinary(self, node: Binary):
        return self.visit(node.exp1), self.visit(node.exp2)

    def visitIfStatement(self, node: IfStatement):
        for b in node.branches:
            self.visitBranch(b)
    
    def visitSliceStatement(self, node: SliceStatement):
        self.visitSlice(node.slice)

    def visitSlice(self, node: Slice):
        for o in node.exps:
            self.visit(o) 

    def visitBranch(self, node: Branch):
        if node.cond:
            self.visit(node.cond)
        self.visit(node.block)


class AstDumper(AstVisitor):
    def __init__(self, prefix="") -> None:
        super().__init__()
        self.prefix = prefix
    
    def visitModule(self, node: AstModule):
        print(self.prefix + "Module:")
        self.inc_indent()
        self.visitBlock(node.block)
        self.dec_indent()
    
    def visitBlock(self, node: Block):
        print(self.prefix + "Block:")
        self.inc_indent()
        for o in node.stmts:
            ret = self.visit(o)
            if ret:
                print(self.prefix + f"{ret}")
    
    def visitBlockEnd(self, node: BlockEnd):
        self.dec_indent()

    def visitFunctionDef(self, node: FunctionDef):
        print(self.prefix + f"Function define {node.name}")
        self.inc_indent()
        self.visitSignature(node.signature)
        self.visitBlock(node.block)
        self.dec_indent()
    
    def visitSignature(self, node: Signature):
        print(self.prefix + f"Signature")
        self.inc_indent()
        print(self.prefix + f"{self.visitParameterList(node.param_list)}")
        self.dec_indent()
    
    def visitParameterList(self, node: ParameterList):
        return [self.visitParamentDef(p) for p in node.params]

    def visitParamentDef(self, node: ParamentDef):
        data = f"<Parameter {node.name}"
        if node.type:
            data += f": {node.type}"
        if node.init:
            data += f" = {self.visit(node.init)}"
        data += ">"
        return data

    def visitFunctionCall(self, node: FunctionCall):
        args = self.visitArgumentList(node.arg_list)
        ref_str = "(resolved)" if node.sym else "(not resolved)"
        return f"Function Call {node.name}, arg_list: {args}  {ref_str}"
    
    def visitArgumentList(self, node: ArgumentList):
        return [self.visit(o) for o in node.args]
    
    def visitPositionalArgument(self, node: PositionalArgument):
        return f'<Arg {node.index} = {self.visit(node.value)}>'

    def visitKeywordArgument(self, node: KeywordArgument):
        return f'<Arg {node.name} = {self.visit(node.value)}>'

    def visitReturnStatement(self, node: ReturnStatement):
        return f"Return {self.visit(node.ret) if node.ret else None}"
    
    def visitVariableDef(self, node: VariableDef):
        init = self.visit(node.init)
        return f"Variable define {node.name}, init: {init}"

    def visitBooleanLiteral(self, node: BooleanLiteral):
        return "True" if node.value else "False"

    def visitNoneLiteral(self, node: NoneLiteral):
        return "None"

    def visitVariable(self, node: Variable):
        ref_str = "(resolved)" if node.sym else "(not resolved)"
        return f'Variable {node.name} {ref_str}'

    def visitImportStatement(self, node: ImportStatement):
        msg = f"Import {node.package}"
        if node.alias:
            msg += f" as {node.alias}"
        return msg
    
    def visitBinary(self, node: Binary):
        return f"<Binary {node.op}: {self.visit(node.exp1)}, {self.visit(node.exp2)}>"
    
    def visitUnary(self, node: Unary):
        return self.visit(node.exp)

    def visitIfStatement(self, node: IfStatement):
        print(self.prefix + "If Statement")
        self.inc_indent()
        for b in node.branches:
            self.visitBranch(b)
        self.dec_indent()

    def visitBranch(self, node: Branch):
        print(self.prefix + ("Else Branch" if node.is_else_branch else "Then Branch"))
        self.inc_indent()
        if not node.is_else_branch:
            print(self.prefix + f"Condition: {self.visit(node.cond)}")
        self.visitBlock(node.block)
        self.dec_indent()
    
    def visitSliceStatement(self, node: SliceStatement):
        return f"Slice {node.name} by {self.visitSlice(node.slice)}"

    def visitSlice(self, node: Slice):
        slices = [str(self.visit(o)) for o in node.exps]
        if node.omitted_first_dim:
            slices = [''] + slices
        if node.omitted_last_dim:
            slices.append('')
        return ':'.join(slices)

    def inc_indent(self):
        self.prefix += "  "

    def dec_indent(self):
        self.prefix = self.prefix[:-2]
