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

from lexer import *
from symbolic import FunctionSymbol, VariableSymbol


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


class ExpressionStatement(Statement):
    def __init__(self, exp: Expression) -> None:
        super().__init__()
        self.exp = exp
    
    def accept(self, visitor):
        return visitor.visitExpressionStatement(self)


class Block(Statement):
    def __init__(self, stmts) -> None:
        super().__init__()
        self.stmts = stmts

    def accept(self, visitor):
        return visitor.visitBlock(self)


class AstModule(AstNode):
    def __init__(self, block: Block) -> None:
        self.block = block

    def accept(self, visitor):
        return visitor.visitModule(self)


class FunctionDecl(Statement):
    def __init__(self, name: str, block: Block) -> None:
        super().__init__()
        self.name = name
        self.block = block
    
    def accept(self, visitor):
        return visitor.visitFunctionDecl(self)


class FunctionCall(Expression):
    def __init__(self, name: str, args: list = [], sym: FunctionSymbol = None) -> None:
        super().__init__()
        self.name = name
        self.args = args
        self.sym = sym
    
    def accept(self, visitor):
        return visitor.visitFunctionCall(self)


class VariableDecl(Statement):
    def __init__(self, name: str, type, init: ExpressionStatement) -> None:
        super().__init__()
        self.name = name
        self.type = type
        self.init = init
    
    def accept(self, visitor):
        return visitor.visitVariableDecl(self)


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


class NoneLiteral(Expression):
    def __init__(self) -> None:
        super().__init__()
        self.value = None
    
    def accept(self, visitor):
        return visitor.visitNoneLiteral(self)


class ReturnStatement(Statement):
    def __init__(self, ret: str = "") -> None:
        super().__init__()
        self.ret = ret
    
    def accept(self, visitor):
        return visitor.visitReturnStatement(self)


class EmptyStatement(Statement):
    def __init__(self) -> None:
        super().__init__()
    
    def accept(self, visitor):
        return visitor.visitEmptyStatement(self)


class AstVisitor(ABC):
    def visit(self, node: AstNode):
        return node.accept(self)

    @abstractmethod
    def visitModule(self, node: AstModule):
        return self.visitBlock(node.block)

    @abstractmethod
    def visitBlock(self, node: Block):
        ret = None
        for o in node.stmts:
            ret = self.visit(o)
            if ret == "return":
                break
        return ret

    @abstractmethod
    def visitFunctionDecl(self, node: FunctionDecl):
        return self.visitBlock(node.block)

    @abstractmethod
    def visitFunctionCall(self, node: FunctionCall):
        pass

    @abstractmethod
    def visitReturnStatement(self, node: ReturnStatement):
        return "return"

    @abstractmethod
    def visitExpressionStatement(self, node: ExpressionStatement):
        return self.visit(node.exp)

    @abstractmethod
    def visitVariableDecl(self, node: VariableDecl):
        if node.init:
            return self.visit(node.init)
    
    @abstractmethod
    def visitVariable(self, node: Variable):
        pass

    @abstractmethod
    def visitStringLiteral(self, node: StringLiteral):
        return node.data

    @abstractmethod
    def visitIntegerLiteral(self, node: IntegerLiteral):
        return node.value

    @abstractmethod
    def visitDecimalLiteral(self, node: DecimalLiteral):
        return node.value

    @abstractmethod
    def visitNoneLiteral(self, node: NoneLiteral):
        return None

    @abstractmethod
    def visitEmptyStatement(self, node: EmptyStatement):
        pass


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
            self.visit(o)
        self.dec_indent()

    def visitFunctionDecl(self, node: FunctionDecl):
        print(self.prefix + f"Function Decl {node.name}")
        self.inc_indent()
        self.visitBlock(node.block)
        self.dec_indent()
    
    def visitFunctionCall(self, node: FunctionCall):
        args = [self.visit(o) for o in node.args]
        print(self.prefix + f"Function Call {node.name}, args: {args}")
    
    def visitReturnStatement(self, node: ReturnStatement):
        print(self.prefix + f"Return {node.ret}")

    def visitExpressionStatement(self, node: ExpressionStatement):
        return super().visitExpressionStatement(node)

    def visitVariableDecl(self, node: VariableDecl):
        init = self.visitExpressionStatement(node.init)
        print(self.prefix + f"Variable Decl {node.name}, init: {init}")

    def visitDecimalLiteral(self, node: DecimalLiteral):
        return super().visitIntegerLiteral(node)

    def visitIntegerLiteral(self, node: IntegerLiteral):
        return super().visitIntegerLiteral(node)

    def visitNoneLiteral(self, node: NoneLiteral):
        return "None"

    def visitStringLiteral(self, node: StringLiteral):
        return super().visitStringLiteral(node)
    
    def visitVariable(self, node: Variable):
        return f'<Variable {node.name}>'

    def visitEmptyStatement(self, node: EmptyStatement):
        return super().visitEmptyStatement(node)

    def inc_indent(self):
        self.prefix += "  "

    def dec_indent(self):
        self.prefix = self.prefix[:-2]
