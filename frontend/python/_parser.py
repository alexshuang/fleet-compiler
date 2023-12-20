# ===- _parser.py -------------------------------------------------------------
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

# from frontend.python.lexer import *
# from frontend.python.ast import *
from lexer import *
from ast import *


class Parser:
    '''
    module = statementList
    statementList = statement*
    statement = functionDecl | functionCall
    funcitonDecl = def Identifier ( ) :
    functionCall = Identifier ( args ) terminator
    terminator = \n | ;
    args = Empty | stringLiteral (, stringLiteral)?
    stringLiteral = " StringLiteral? "
    '''
    def __init__(self, data="") -> None:
        self.tokenizer = Tokenizer(data)
    
    def parse_module(self):
        stmts = []
        while self.tokenizer.peak().kind != TokenKind.EOF:
            stmts.append(self.parse_statement())
        return ASTModule(Block(stmts))
    
    def parse_statement(self):
        t = self.tokenizer.peak()
        if t.data == "def":
            return self.parse_function_decl()
        elif t.kind == TokenKind.Identifier:
            return self.parse_identifier()
        elif t.kind == TokenKind.Terminator:
            self.tokenizer.next() # skip it
            return self.parse_statement()
        else:
            self.raise_error(f"Unrecognized token {t.data}, kind {t.kind}")
    
    def parse_function_decl(self):
        self.tokenizer.next() # skip def
        t = self.tokenizer.peak()
        if t.kind == TokenKind.Identifier:
            func_name = self.tokenizer.next().data # skip identifier
            t = self.tokenizer.peak()
            if t.data == '(':
                self.tokenizer.next() # skip (
                t = self.tokenizer.peak()
                if t.data == ')':
                    self.tokenizer.next() # skip )
                    t = self.tokenizer.peak()
                    if t.data == ':':
                        self.tokenizer.next() # skip :
                        func_body = self.parse_block()
                    else:
                        self.raise_error(f"Expect got ':' here, not {t.data}")
                else:
                    self.raise_error(f"Expect got ')' here, not {t.data}")
            else:
                self.raise_error(f"Expect got '(' here, not {t.data}")
        else:
            self.raise_error(f"Expect got identifier here, not {t.data}")
        return FunctionDecl(func_name, func_body)
    
    def parse_block(self):
        stmts = []
        while self.tokenizer.peak().kind != TokenKind.EOF:
            stmt = self.parse_statement()
            stmts.append(stmt)
            if isinstance(stmt, ReturnStatement):
                break
        return Block(stmts)

    def parse_identifier(self):
        name = self.tokenizer.next().data # skip identifier
        if name == "return":
            return self.parse_return()
        else:
            t = self.tokenizer.peak()
            if t.data == '(':
                self.tokenizer.next() # skip (
                t = self.tokenizer.peak()
                args = self.parse_args() if t.data != ')' else []
                t = self.tokenizer.peak()
                if t.data == ')':
                    self.tokenizer.next() # skip )
                    t = self.tokenizer.peak()
                    if t.kind == TokenKind.Terminator or t.kind == TokenKind.EOF:
                        self.tokenizer.next() # skip terminator
                        return FunctionCall(name, args)
                    else:
                        self.raise_error(f"Expect got newline or ';' here, not {t.data}")
                else:
                    self.raise_error(f"Expect got ')' here, not {t.data}")
            else:
                self.raise_error(f"Expect got '(' here, not {t.data}")

    def parse_args(self):
        args = []
        t = self.tokenizer.peak()
        while t.kind != TokenKind.EOF and t.data != ')':
            t = self.tokenizer.next()
            if t.kind == TokenKind.StringLiteral:
                args.append(t.data)
            t = self.tokenizer.peak()
            if t.data != ')':
                if t.data == ',':
                    t = self.tokenizer.next()
                else:
                    self.raise_error(f"Expect got ',' here, not {t.data}")
    
    def parse_return(self):
        t = self.tokenizer.peak()
        print(f"{t.kind}@{t.data}")
        if t.kind == TokenKind.Terminator or t.kind == TokenKind.EOF:
            self.tokenizer.next() # skip terminator
        else:
            self.raise_error(f"Expect got newline or ';' here, not {t.kind}@{t.data}")
        return ReturnStatement()

    def raise_error(self, msg):
        raise ValueError(f"{self.tokenizer.location_str} : {msg}")
