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
from symbolic import FunctionSymbol


class AstNode(ABC):
    @abstractmethod
    def accept(self, visitor):
        pass


class Statement(AstNode):
    def __init__(self) -> None:
        super().__init__()


class Block(Statement):
    def __init__(self, stmts) -> None:
        super().__init__()
        self.stmts = stmts

    def accept(self, visitor):
        visitor.visitBlock(self)


class AstModule(AstNode):
    def __init__(self, block: Block) -> None:
        self.block = block

    def accept(self, visitor):
        visitor.visitModule(self)


class FunctionDecl(Statement):
    def __init__(self, name: str, block: Block) -> None:
        super().__init__()
        self.name = name
        self.block = block
    
    def accept(self, visitor):
        visitor.visitFunctionDecl(self)


class FunctionCall(Statement):
    def __init__(self, name: str, args: list = [], sym: FunctionSymbol = None) -> None:
        super().__init__()
        self.name = name
        self.args = args
        self.sym = sym
    
    def accept(self, visitor):
        visitor.visitFunctionCall(self)


class ReturnStatement(Statement):
    def __init__(self, ret: str = "") -> None:
        super().__init__()
        self.ret = ret
    
    def accept(self, visitor):
        visitor.visitReturnStatement(self)


class AstVisitor(ABC):
    def visit(self, node: AstNode):
        node.accept(self)
    
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
        return
        
    @abstractmethod
    def visitReturnStatement(self, node: ReturnStatement):
        return "return"
    

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
        print(self.prefix + f"Function Call {node.name}, args: {node.args}")
    
    def visitReturnStatement(self, node: ReturnStatement):
        print(self.prefix + f"Return {node.ret}")

    def inc_indent(self):
        self.prefix += "  "

    def dec_indent(self):
        self.prefix = self.prefix[:-2]
