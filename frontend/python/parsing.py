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

from lexer import *
from syntax import *
from dtype import *
from indentation import *


class Parser:
    '''
    module = block
    block = statementList
    statementList = statement*
    statement = indentation (functionDecl | functionCall | returnStatement | variableDecl | expressionStatement | emptyStatement)
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
    expressionStatement = expression terminator
    expression = assignment
    assignment = binary (assignmentOp binary)*
    binary = unary (binOp unary)*
    unary = primary
    primary = StringLiteral | IntegerLiteral | DecimalLiteral | NoneLiteral | functionCall
    binOp = '+' | '-' | '*' | '/'
    assignmentOp = '='
    stringLiteral = '"' StringLiteral? '"'
    '''
    def __init__(self, data="") -> None:
        self.tokenizer = Tokenizer(data)
        self.indentation = Indentation()
        self.checkpoint = CheckPoint(self.tokenizer)
    
    def parse_module(self):
        return AstModule(self.parse_block())
    
    def parse_statement(self):
        '''
        statement = indentation (functionDecl | functionCall | returnStatement | variableDecl | expressionStatement | emptyStatement)
        '''
        t = self.tokenizer.peak()
        if t.kind != TokenKind.Indentation:
            self.raise_error(f"Expect indentation here, not {t.kind}@{t.data}")
        if self.indentation.match(t.data):
            self.tokenizer.next() # skip indent
        else:
            if self.indentation.match_parent(t.data):
                # implicitly add a return depending on the indentation
                self.tokenizer.next() # skip indent
                return BlockEnd()
            else:
                self.raise_error("Unexpected indentation")

        t = self.tokenizer.peak()
        if t.data == "def":
            ret = self.parse_function_decl()
        elif t.data == "return":
            ret = self.parse_return()
        elif t.kind == TokenKind.Identifier:
            ret = self.parse_identifier()
        elif t.kind == TokenKind.Terminator: # empty statement
            self.tokenizer.next() # skip it
            return EmptyStatement()
        else:
            self.raise_error(f"Unrecognized token {t.data}, kind {t.kind}")
        self.skip_terminator()
        return ret
    
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
                    func_body = self.parse_block()
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
        self.enter()
        stmts = []
        while self.tokenizer.peak().kind != TokenKind.EOF:
            stmt = self.parse_statement()
            if stmt and not isinstance(stmt, EmptyStatement):
                stmts.append(stmt)
            if isinstance(stmt, BlockEnd):
                break
        self.exit()
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
            args = self.parse_function_args()
            ret = FunctionCall(name, args)
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

    def parse_function_args(self):
        self.tokenizer.next() # skip (
        t = self.tokenizer.peak()
        args = self.parse_args() if t.data != ')' else []
        t = self.tokenizer.peak()
        if t.data == ')':
            self.tokenizer.next() # skip )
        else:
            self.raise_error(f"Expect got ')' here, not {t.data}")
        return args

    def parse_expression_statement(self):
        self.tokenizer.next() # skip =
        exp = self.parse_expression()
        return ExpressionStatement(exp)

    def parse_expression(self):
        '''
        expression = assignment
        assignment = binary (assignmentOp binary)*
        binary = unary (binOp unary)*
        unary = primary
        primary = StringLiteral | IntegerLiteral | DecimalLiteral | NoneLiteral | functionCall
        '''
        return self.parse_assignment()
    
    def parse_assignment(self):
        return self.parse_binary()

    def parse_binary(self):
        return self.parse_unary()

    def parse_unary(self):
        return self.parse_primary()
    
    def parse_primary(self):
        t = self.tokenizer.next()
        if t.kind == TokenKind.StringLiteral:
            return StringLiteral(t.data)
        elif t.kind == TokenKind.IntegerLiteral:
            return IntegerLiteral(int(t.data))
        elif t.kind == TokenKind.DecimalLiteral:
            return DecimalLiteral(float(t.data))
        elif t.kind == TokenKind.NoneLiteral:
            return NoneLiteral()
        elif t.kind == TokenKind.Identifier:
            if self.tokenizer.peak().data == '(':
                name = t.data
                args = self.parse_function_args()
                return FunctionCall(name, args)
            else:
                return Variable(t.data)
        else:
            self.raise_error(f"Unsupport primary {t.kind}@{t.data}")

    def parse_args(self):
        args = []
        t = self.tokenizer.peak()
        while t.kind != TokenKind.EOF and t.data != ')':
            args.append(self.parse_expression())
            t = self.tokenizer.peak()
            if t.data != ')':
                if t.data == ',':
                    t = self.tokenizer.next()
                else:
                    self.raise_error(f"Expect got ',' here, not {t.data}")
        return args
    
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

    def enter(self):
        # enter new indent
        self.last_indentation = self.indentation
        self.indentation = Indentation(self.indentation)

    def exit(self):
        # back to last indent
        self.indentation = self.last_indentation
        
    def raise_error(self, msg):
        raise SyntaxError(f"{self.tokenizer.location_str} : {msg}")
