# ===- parsing.py -------------------------------------------------------------
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

from .lexer import *
from .syntax import *
from .dtype import *
from .indentation import *

class PositionalArgumentIndex:
    def __init__(self, parent=None) -> None:
        self.parent = parent
        self.value = 0
    
    def get(self):
        return self.value
    
    def inc(self):
        self.value += 1


OpPriority = {
    Op.Assign: 2,
    Op.GT: 4,
    Op.GE: 4,
    Op.LT: 4,
    Op.LE: 4,
    Op.EQ: 4,
    Op.NE: 4,
    Op.Plus: 5,
    Op.Minus: 5,
    Op.Multiply: 6,
    Op.Divide: 6,
    Op.AT: 7,
    Op.Power: 10,
}

def get_op_priority(op):
    return OpPriority[op] if isinstance(op, Op) and op in OpPriority else -1


class Parser:
    '''
    module = block
    block = statementList
    statementList = statement*
    statement = indentation (functionDecl | functionCall | returnStatement | \
        variableDecl | expressionStatement | emptyStatement | importStatement)
    indentation = ' '*
    funcitonDecl = 'def' Identifier signature ':'
    functionCall = Identifier '(' args ')' terminator
    signature = '(' parameterDecl? (',' parameterDecl)? ')' (-> Identifier)?
    parameterDecl = Identifier typeAnnotation? ('=' expressionStatement)?
    returnStatement = 'return' expressionStatement
    terminator = '\n' | ';'
    args = expression (',' expression)*
    variableDecl = Identifier typeAnnotation? '=' expressionStatement
    typeAnnotation = ':' typeName
    typeName = StringLiteral
    emptyStatement = terminator
    importStatement = 'import' package ('as' Identifier)? terminator
    package = Identifier ('.' Identifier)*
    expressionStatement = expression terminator
    expression = assignment
    assignment = binary (assignmentOp binary)*
    binary = unary (binOp unary)*
    unary = primary
    primary = StringLiteral | IntegerLiteral | DecimalLiteral | NoneLiteral |
        BooleanLiteral | functionCall | '(' expression ')'
    binOp = '+' | '-' | '*' | '/'
    assignmentOp = '='
    stringLiteral = '"' StringLiteral? '"'
    '''
    def __init__(self, data="") -> None:
        self.tokenizer = Tokenizer(data)
        self.indentation = None
        self.pos_arg_idx = None
        self.not_allow_indent = True # not allow indent in main phase
    
    def parse_module(self):
        return AstModule(self.parse_block())
    
    def parse_statement(self):
        '''
        statement = newline | indentation (functionDecl | functionCall | returnStatement | variableDecl | expressionStatement | emptyStatement)
        '''
        t = self.tokenizer.peak()
        if t.kind == TokenKind.Newline:
            self.tokenizer.next() # skip \n
            return self.parse_statement()
        
        # If it is not an indentation, then should go directly to main_phase,
        # which not_allow_indent = True. Otherwise, it will rollback indent and 
        # determine the boundary of the block
        if t.kind != TokenKind.Indentation:
            self.indentation = None
            if not self.not_allow_indent:
                self.not_allow_indent = True
                return BlockEnd()
        else:
            if self.not_allow_indent:
                self.raise_error(f"Unexpected indentation")
            elif self.indentation.match(t.data):
                self.tokenizer.next() # skip indent
                t = self.tokenizer.peak()
            elif self.indentation.match_parents(t.data):
                self.exit_indent()
                return BlockEnd()
            else:
                self.raise_error(f"Unexpected indentation")
        
        if t.data == "def":
            ret = self.parse_function_decl()
        elif t.data == "return":
            ret = self.parse_return()
        elif t.data == "import":
            ret = self.parse_import()
        elif t.kind == TokenKind.Identifier:
            ret = self.parse_identifier()
        elif t.kind == TokenKind.Terminator: # empty statement
            self.tokenizer.next() # skip it
            return EmptyStatement()
        elif t.kind == TokenKind.EOF: # EOF
            return EmptyStatement()
        else:
            self.raise_error(f"Unrecognized token {t.data}, kind {t.kind}")
        self.skip_terminator()
        return ret
    
    def parse_import(self):
        alias = ""
        self.tokenizer.next() # skip import
        t = self.tokenizer.peak()
        if t.kind == TokenKind.Identifier:
            pkg = self.tokenizer.next().data
            if self.tokenizer.peak().data == 'as':
                self.tokenizer.next() # skip as
                alias = self.tokenizer.next().data
            return ImportStatement(pkg, alias)
        else:
            self.raise_error(f"Expect got identifier here, not {t.data}")
    
    def parse_function_decl(self):
        self.tokenizer.next() # skip def
        t = self.tokenizer.peak()
        if t.kind == TokenKind.Identifier:
            func_name = self.tokenizer.next().data # skip identifier
            t = self.tokenizer.peak()
            if t.data == '(':
                signature = self.parse_signature()
                t = self.tokenizer.peak()
                if t.data == ':':
                    self.tokenizer.next() # skip :
                    self.skip_terminator() # skip nl
                    self.enter_indent()
                    func_body = self.parse_block()
                    # self.exit_indent()
                else:
                    self.raise_error(f"Expect got ':' here, not {t.data}")
            else:
                self.raise_error(f"Expect got '(' here, not {t.data}")
        else:
            self.raise_error(f"Expect got identifier here, not {t.data}")
        return FunctionDecl(func_name, signature, func_body)
    
    def parse_signature(self):
        self.tokenizer.next() # skip (
        param_list = self.parse_parameter_list()
        t = self.tokenizer.peak()
        if t.data == ')':
            self.tokenizer.next() # skip )
        else:
            self.raise_error(f"Expect got ')' here, not {t.data}")
        return Signature(param_list)

    def parse_parameter_list(self):
        params = []
        t = self.tokenizer.peak()
        while t.kind != TokenKind.EOF and t.data != ')':
            params.append(self.parse_parameter_decl())
            t = self.tokenizer.peak()
            if t.data != ')':
                if t.data == ',':
                    t = self.tokenizer.next()
                else:
                    self.raise_error(f"Expect got ',' or ')' here, not {t.data}")
        return ParameterList(params)
    
    def parse_parameter_decl(self):
        # parameterDecl = Identifier typeAnnotation? ('=' expressionStatement)?
        t = self.tokenizer.peak()
        if t.kind != TokenKind.Identifier:
            self.raise_error(f"Expect got Identifier here, not {t.data}")

        name = t.data
        type = init = None
        self.tokenizer.next() # skip identifier
        
        t = self.tokenizer.peak()
        if t.data == ":":
            type = self.parse_variable_type()
            t = self.tokenizer.peak()
        if t.data == "=":
            init = self.parse_expression_statement()

        return ParameterDecl(name, type, init)
        
    def parse_block(self):
        stmts = []
        while self.tokenizer.peak().kind != TokenKind.EOF:
            stmt = self.parse_statement()
            if stmt and not isinstance(stmt, EmptyStatement):
                stmts.append(stmt)
            if isinstance(stmt, BlockEnd):
                break
        return Block(stmts)

    def parse_identifier(self):
        '''
        functionCall = Identifier '(' args ')' terminator
        variableDecl = Identifier typeAnnotation? '=' expressionStatement terminator
        '''
        name = self.tokenizer.next().data # skip identifier
        t = self.tokenizer.peak()
        if t.data == ":":
            type = self.parse_variable_type()
            t = self.tokenizer.peak()
            if t.data == "=":
                init = self.parse_expression_statement()
                ret = VariableDecl(name, type, init)
            else:
                ret = Variable(name, type)
        elif t.data == "=":
            init = self.parse_expression_statement()
            ret = VariableDecl(name, None, init)
        elif t.data == "(":
            ret = self.parse_function_call(name)
        else:
            self.raise_error(f"Unsupport statement which start with {t.data}")
        self.skip_terminator()
        return ret 

    def parse_variable_type(self):
        self.tokenizer.next() # skip :
        t = self.tokenizer.peak()
        if t.kind == TokenKind.Identifier:
            t = self.tokenizer.next()
            if t.data in dtype_map:
                return dtype_map[t.data.lower()]
            else:
                self.raise_error(f"Unsupport data type {t.data}")
        else:
            self.raise_error(f"Invalid data type {t.data}")
    
    def parse_function_call(self, name: str):
        arg_list = self.parse_arg_list()
        return FunctionCall(name, arg_list)

    def parse_expression_statement(self):
        self.tokenizer.next() # skip =
        exp = self.parse_expression()
        return ExpressionStatement(exp)

    def parse_expression(self):
        '''
        expression = assignment
        '''
        return self.parse_assignment()
    
    def parse_assignment(self):
        '''
        assignment = binary (assignmentOp binary)*
        '''
        op_list, exp_list = [], []
        exp_list.append(self.parse_binary(get_op_priority(Op.Assign)))
        while self.tokenizer.peak().data == '=':
            op_list.append(self.tokenizer.next().code) # skip =
            exp_list.append(self.parse_binary(get_op_priority(Op.Assign)))
        assert(len(exp_list) > 0)
        exp = exp_list.pop()
        assert(len(exp_list) == len(op_list))
        while len(exp_list) > 0:
            exp = Binary(op_list.pop(), exp_list.pop(), exp)
        return exp

    def parse_binary(self, prev_pri: int):
        '''
        binary = unary (binOp unary)*
        '''
        exp1 = self.parse_unary()
        t = self.tokenizer.peak()
        pri = get_op_priority(t.code)
        while t.kind == TokenKind.OP and pri > prev_pri:
            self.tokenizer.next() # skip op
            exp2 = self.parse_binary(pri)
            exp1 = Binary(t.code, exp1, exp2)
            t = self.tokenizer.peak()
            pri = get_op_priority(t.code)
        return exp1

    def parse_unary(self):
        '''
        unary = primary
        '''
        return Unary(self.parse_primary())
    
    def parse_primary(self):
        '''
        primary = StringLiteral | IntegerLiteral | DecimalLiteral | NoneLiteral |
            BooleanLiteral | functionCall | '(' expression ')'
        '''
        ret = None
        t = self.tokenizer.peak()
        if t.kind == TokenKind.StringLiteral:
            ret = StringLiteral(t.data)
        elif t.kind == TokenKind.IntegerLiteral:
            ret = IntegerLiteral(int(t.data))
        elif t.kind == TokenKind.DecimalLiteral:
            ret = DecimalLiteral(float(t.data))
        elif t.kind == TokenKind.BooleanLiteral:
            ret = BooleanLiteral(True if t.data == 'True' else False)
        elif t.kind == TokenKind.NoneLiteral:
            ret = NoneLiteral()
        elif t.data == '(':
            self.tokenizer.next() # skip '('
            ret = self.parse_expression()
            assert(self.tokenizer.peak().data == ')')
        
        if ret:
            self.tokenizer.next()
            return ret
        
        elif t.kind == TokenKind.Identifier:
            self.tokenizer.next()
            if self.tokenizer.peak().data == '(':
                return self.parse_function_call(t.data)
            else:
                return Variable(t.data)
        else:
            self.raise_error(f"Unsupport primary {t.kind}@{t.data}")

    def parse_arg_list(self):
        args = []
        self.pos_arg_idx = PositionalArgumentIndex(self.pos_arg_idx)
        self.tokenizer.next() # skip (
        t = self.tokenizer.peak()
        while t.kind != TokenKind.EOF and t.data != ')':
            args.append(self.parse_argument())
            t = self.tokenizer.peak()
            if t.data != ')':
                if t.data == ',':
                    t = self.tokenizer.next()
                else:
                    self.raise_error(f"Expect got ',' here, not {t.data}")
        t = self.tokenizer.peak()
        if t.data == ')':
            self.tokenizer.next() # skip )
            self.pos_arg_idx = self.pos_arg_idx.parent
        else:
            self.raise_error(f"Expect got ')' here, not {t.data}")
        return ArgumentList(args)

    def parse_argument(self):
        '''
        positionalArgumentList = positionalArgument (',' positionalArgument)*
        positionalArgument = expression
        keywordArgumentList = keywordArgument (',' keywordArgument)*
        keywordArgument = Identifier '=' expression
        '''
        ret = None
        self.tokenizer.save_checkpoint()
        t = self.tokenizer.next()
        if t.kind == TokenKind.Identifier and self.tokenizer.peak().data == "=":
            self.tokenizer.next() # skip =
            value = self.parse_expression()
            ret = KeywordArgument(t.data, value)
        else:
            self.tokenizer.load_checkpoint()
            value = self.parse_expression()
            ret = PositionalArgument(self.pos_arg_idx.get(), value)
        self.pos_arg_idx.inc()
        return ret

    def parse_return(self):
        ret = None
        self.tokenizer.next() # skip return
        if self.tokenizer.peak().kind != TokenKind.Terminator:
            ret = self.parse_return_value()
        self.skip_terminator()
        return ReturnStatement(ret)

    def parse_return_value(self):
        return self.parse_expression()

    def skip_terminator(self):
        t = self.tokenizer.peak()
        if t.kind == TokenKind.Terminator:
            self.tokenizer.next() # skip terminator

    def enter_indent(self):
        # enter new indent
        self.indentation = Indentation(self.indentation)
        self.not_allow_indent = False

    def exit_indent(self):
        # back to last indent
        self.indentation = self.indentation.parent
        
    def raise_error(self, msg):
        raise SyntaxError(f"{self.tokenizer.location_str} : {msg}")
