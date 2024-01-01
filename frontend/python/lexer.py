# ===- lexer.py -------------------------------------------------------------
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

from enum import Enum, auto


def is_boolean(data):
    return data in ['True', 'False']

def is_keyword(data):
    return data in ['def']

def is_terminator(ch):
    return ch == '\n' or ch == ';'

def is_operator(ch):
    return ch in ['+', '-', '*', '/', '%', '^', '&', '=', '!', '@']

def is_alpha(ch):
    return (ch >= 'a' and ch <= 'z') or (ch >= 'A' and ch <= 'Z')

def is_digit(ch):
    return ch >= '0' and ch <= '9'

def is_separator(ch):
    return ch in [' ', '(', ')', '#', ':', '[', ']', ',']


class CharStream:
    def __init__(self, data="") -> None:
        self.data = data
        self.pos = 0
        self.len = len(data)
        self.line = 1
        self.col = 1

    def next(self):
        if self.eof():
            return ''
        ch = self.data[self.pos]
        self.pos += 1
        self.col += 1
        if ch == "\n":
            self.line += 1
            self.col = 1
        return ch

    def peak(self, offset=0):
        if self.eof() or self.pos + offset >= self.len:
            return ''
        return self.data[self.pos + offset]

    def eof(self):
        return self.pos >= self.len
    
    @property
    def location_str(self):
        return f"@(line: {self.line}, col: {self.col})"
    

class TokenKind(Enum):
    Keyword = 0
    Identifier = auto()
    Separator = auto()
    StringLiteral = auto()
    IntegerLiteral = auto()
    DecimalLiteral = auto()
    NoneLiteral = auto()
    BooleanLiteral = auto()
    Terminator = auto()
    Indentation = auto()
    Newline = auto()
    OP = auto()
    EOF = auto()


class Op(Enum):
    Assign = 0               # =

    Plus = auto()      # +
    Minus = auto()     # -
    Multiply = auto()  # *
    Divide = auto()    # /
    AT = auto()        # @
    Power = auto()     # **

    GT = auto()      # >
    GE = auto()      # >=
    LT = auto()      # <
    LE = auto()      # <=
    EQ = auto()      # ==
    NE = auto()      # !=


op_str2code = {
    '=': Op.Assign,
    '+': Op.Plus,
    '-': Op.Minus,
    '*': Op.Multiply,
    '/': Op.Divide,
    '@': Op.AT,
    '**': Op.Power,
    '>': Op.GT,
    '>=': Op.GE,
    '<': Op.LT,
    '<=': Op.LE,
    '==': Op.EQ,
    '!=': Op.NE,
}


class Token:
    def __init__(self, kind=TokenKind.EOF, data="", code=-1) -> None:
        self.kind = kind
        self.data = data
        self.code = code
    

class Tokenizer:
    def __init__(self, data="") -> None:
        self.stream = CharStream(data)
        self.current_token = None
        self.start_of_line = True
        self.pos = 0
        self.line = 1
        self.col = 1
        self.checkpoint = {}

    def next(self):
        if self.current_token == None:
            self.current_token = self.get_token()
        ret = self.current_token
        self.update_cursor()
        self.current_token = self.get_token()
        return ret

    def peak(self):
        if self.current_token == None:
            self.current_token = self.get_token()
        return self.current_token

    def get_token(self):
        if self.stream.eof():
            return Token()

        if self.start_of_line == True:
            if self.stream.peak() == '\n':
                self.stream.next()
                return Token(TokenKind.Newline)

            self.start_of_line = False
            if self.stream.peak() == ' ':
                return self.parse_indentation()
        else:
            self.skip_white_space()

        ch = self.stream.peak()
        if ch == '#':
            self.skip_single_comment()
            return self.get_token()
        elif ch == '\'' and self.stream.peak(1) == '\'' and self.stream.peak(2) == '\'':
            self.skip_comments()
            return self.get_token()

        # parse token 
        if is_alpha(ch):
            return self.parse_identifier()
        elif is_digit(ch) or (ch == '.' and is_digit(self.stream.peak(1))) \
            or (ch == '-' and is_digit(self.stream.peak(1))):
            return self.parse_digit()
        elif ch == '\n':
            return self.parse_newline()
        elif is_operator(ch):
            return self.parse_operator()
        elif is_separator(ch):
            return self.parse_separator()
        elif is_terminator(ch):
            return self.parse_terminator()
        elif ch == '"' or ch == '\'':
            return self.parse_string_literal()
        else:
            self.raise_error(f"Unrecognized token which start with {ch}")

    def parse_operator(self):
        data = self.stream.next()
        if data == '=':
            if self.stream.peak() in ['=']:
                data += self.stream.next()
        elif data == '*':
            if self.stream.peak() in ['*']:
                data += self.stream.next()
        elif data == '>':
            if self.stream.peak() in ['=']:
                data += self.stream.next()
        elif data == '<':
            if self.stream.peak() in ['=']:
                data += self.stream.next()
        elif data == '!':
            if self.stream.peak() != '=':
                self.raise_error(f"Unsupported operator {data + self.stream.peak()} here")
        return Token(TokenKind.OP, data, op_str2code[data])
        
    def parse_indentation(self):
        data = ""
        while self.stream.peak() == ' ':
            data += self.stream.next()
        return Token(TokenKind.Indentation, data)
        
    def parse_newline(self):
        self.start_of_line = True
        self.stream.next() # skip \n
        return Token(TokenKind.Terminator, "")

    def parse_string_literal(self):
        quote = self.stream.next() # skip left quote
        data = ""
        while not self.stream.eof() and self.stream.peak() != quote:
            data += self.stream.next()
        
        if self.stream.eof():
            self.raise_error(f"Should end with {quote} here")
        
        self.stream.next() # skip right quote
        return Token(TokenKind.StringLiteral, data)

    def parse_identifier(self):
        def is_valid(ch):
            return is_alpha(ch) or is_digit(ch) or ch == '_' or ch == '.'
        
        data = ""
        ch = self.stream.peak()
        while not self.stream.eof() and not (is_separator(ch) or is_operator(ch) or is_terminator(ch)):
            if not is_valid(ch):
                self.raise_error(f"Invalid identifier {data} with char {ch}")
            data += ch
            self.stream.next()
            ch = self.stream.peak()
        
        kind = TokenKind.Identifier
        if is_keyword(data):
            kind = TokenKind.Keyword
        elif is_boolean(data):
            kind = TokenKind.BooleanLiteral
        elif data == 'None':
            kind = TokenKind.NoneLiteral
            
        return Token(kind, data)

    def parse_digit(self):
        def is_valid(ch):
            return is_digit(ch) or ch == '.' or ch == '-'

        data = ""
        dot_cnt = 0
        ch = self.stream.peak()

        # sign
        if ch == '-':
            data += ch
            self.stream.next()
            ch = self.stream.peak()

        while not self.stream.eof() and not (is_separator(ch) or is_operator(ch) or is_terminator(ch)):
            if not is_valid(ch):
                self.raise_error(f"Invalid number {data}")
            data += ch
            if ch == '.':
                if dot_cnt > 0:
                    self.raise_error(f"Redundant decimal point {data}")
                dot_cnt += 1
            self.stream.next()
            ch = self.stream.peak()
        kind = TokenKind.DecimalLiteral if dot_cnt > 0 else TokenKind.IntegerLiteral
        return Token(kind, data)

    def parse_separator(self):
        ch = self.stream.next()
        return Token(TokenKind.Separator, ch)
    
    def parse_terminator(self):
        ch = self.stream.next()
        return Token(TokenKind.Terminator, ch)

    def skip_white_space(self):
        while self.stream.peak() == ' ':
            self.stream.next()

    def skip_single_comment(self):
        self.stream.next() # skip '#'
        while not self.stream.eof() and self.stream.peak() != '\n':
            self.stream.next()

    def skip_comments(self):
        for _ in range(3): # skip "'''"
            self.stream.next()
        ch1 = self.stream.next()
        ch2 = self.stream.next()
        while not self.stream.eof():
            if self.stream.peak() == '\'' and ch1 == ch2 == '\'':
                self.stream.next()
                return
            else:
                ch1 = ch2
                ch2 = self.stream.next()
        self.raise_error(f"Should end with ''' here")

    def update_cursor(self):
        self.line = self.stream.line
        self.col = self.stream.col
        self.pos = self.stream.pos
    
    def save_checkpoint(self):
        self.checkpoint['pos'] = self.pos
        self.checkpoint['line'] = self.line
        self.checkpoint['col'] = self.col
        self.checkpoint['start_of_line'] = self.start_of_line
    
    def load_checkpoint(self):
        self.stream.pos = self.checkpoint['pos']
        self.stream.line = self.checkpoint['line']
        self.stream.col = self.checkpoint['col']
        self.start_of_line = self.checkpoint['start_of_line']
        self.current_token = None

    @property
    def location_str(self):
        return f"@(line: {self.line}, col: {self.col})"
    
    def raise_error(self, msg):
        raise SyntaxError(f"{self.stream.location_str} : {msg}")
        
