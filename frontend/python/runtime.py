# ===- runtime.py -------------------------------------------------------------
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

from syntax import *


class RetVal:
    def __init__(self, value=None) -> None:
        self.value = value


class StackFrame:
    def __init__(self) -> None:
        self.variables = {} # local variables
        self.ret: RetVal = None # return value

    def update(self, name: str, value):
        self.variables[name] = value

    def get(self, name: str):
        return self.variables[name] if name in self.variables else None
    
    def set_ret(self, ret: RetVal):
        self.ret = ret

    def get_ret(self):
        return self.ret


class Interpreter(AstVisitor):
    def __init__(self) -> None:
        super().__init__()
        self.call_stack = []
    
    def visitModule(self, node: AstModule):
        self.enter()
        return super().visitModule(node)

    def visitBlock(self, node: Block):
        def should_run(node):
            # Instructions that should be executed
            return not isinstance(node, FunctionDecl)

        ret = None
        for o in node.stmts:
            if should_run(o):
                ret = self.visit(o)
                if isinstance(ret, RetVal):
                    break

        if isinstance(ret, RetVal) and ret.value:
            self.set_ret(ret.value)
    
    def visitBlockEnd(self, node: BlockEnd):
        return RetVal()
    
    def visitFunctionCall(self, node: FunctionCall):
        if node.sym:
            self.enter()

            # args
            func = node.sym.node
            params = func.signature.param_list.params
            args = node.arg_list.args
            num_args = len(args)
            num_params = len(params)
            if num_params < num_args:
                raise TypeError(f"Interpreter: Expect {num_params} arguments but got {num_args}")

            names, values = [], []
            for i, arg in enumerate(args):
                names.append(params[i].name if isinstance(arg, PositionalArgument) else arg.name)
                values.append(self.visit(arg.value))
            # args with default-value
            for p in params[num_args:]:
                names.append(p.name)
                values.append(self.visit(p.init))

            # set variable values
            for k, v in zip(names, values):
                self.update_variable_value(k, v)

            # body
            ret = self.visit(node.sym.node)

            self.exit()
            return ret
        else:
            if node.name == "print":
                args = self.visitArgumentList(node.arg_list)
                print(args)
            else:
                raise TypeError(f"Interpreter: Unsupport operation or function {node.name}")
    
    def visitArgumentList(self, node: ArgumentList):
        args = [self.visit(o) for o in node.args if o]
        return args if len(args) > 0 else ""
    
    def visitPositionalArgument(self, node: PositionalArgument):
        return super().visitPositionalArgument(node)

    def visitKeywordArgument(self, node: KeywordArgument):
        return super().visitKeywordArgument(node)

    def visitVariableDecl(self, node: VariableDecl):
        self.update_variable_value(node.name, self.visit(node.init))
        return super().visitVariableDecl(node)
    
    def visitVariable(self, node: Variable):
        return self.get_variable_value(node.name)

    def visitReturnStatement(self, node: ReturnStatement):
        value = self.visit(node.ret) if node.ret else None
        return RetVal(value)

    def visitFunctionDecl(self, node: FunctionDecl):
        return super().visitBlock(node.block)
    
    def visitDecimalLiteral(self, node: DecimalLiteral):
        return super().visitDecimalLiteral(node)
    
    def visitEmptyStatement(self, node: EmptyStatement):
        return super().visitEmptyStatement(node)
    
    def visitExpressionStatement(self, node: ExpressionStatement):
        return super().visitExpressionStatement(node)
    
    def visitIntegerLiteral(self, node: IntegerLiteral):
        return super().visitIntegerLiteral(node)
    
    def visitNoneLiteral(self, node: NoneLiteral):
        return super().visitNoneLiteral(node)

    def visitStringLiteral(self, node: StringLiteral):
        return super().visitStringLiteral(node)
    
    def visitSignature(self, node: Signature):
        return super().visitSignature(node)
    
    def visitParameterList(self, node: ParameterList):
        return super().visitParameterList(node)

    def visitParameterDecl(self, node: ParameterDecl):
        self.update_variable_value(node.name, self.visit(node.init))
        return super().visitParameterDecl(node)
    
    def enter(self):
        self.call_stack.append(StackFrame())
    
    def exit(self):
        self.call_stack.pop()

    def get_variable_value(self, name: str):
        value = None
        for i in range(len(self.call_stack) - 1, -1, -1):
            value = self.call_stack[i].get(name)
            if value:
                break
        if value == None:
            raise NameError(f"name {name} is not defined")
        return value

    def update_variable_value(self, name: str, value):
        frame = self.call_stack[-1]
        if frame:
            frame.update(name, value)

    def set_ret(self, value):
        frame = self.call_stack[-1]
        if frame:
            frame.set_ret(value)
