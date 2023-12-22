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

from enum import Enum


def is_keyword(data):
    return data in ['def', 'if', 'while', 'for', 'with']

def is_terminator(ch):
    return ch == '\n' or ch == ';'

def is_operator(ch):
    return ch in ['+', '-', '*', '/', '%', '^', '&']

def is_alpha(ch):
    return (ch >= 'a' and ch <= 'z') or (ch >= 'A' and ch <= 'Z')

def is_digit(ch):
    return ch >= '0' and ch <= '9'

def is_separator(ch):
    return ch in [' ', '(', ')', '#', ':', '[', ']', '=', ',', '!'] \
        or is_operator(ch) \
        or is_terminator(ch)


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
    Keyword = 1
    Identifier = 2
    Separator = 3
    StringLiteral = 4
    IntegerLiteral = 5
    DecimalLiteral = 6
    NoneLiteral = 7
    Terminator = 8
    Indentation = 9
    EOF = 10


class Token:
    def __init__(self, kind=TokenKind.EOF, data="") -> None:
        self.kind = kind
        self.data = data
    

class Tokenizer:
    def __init__(self, data="") -> None:
        self.stream = CharStream(data)
        self.current_token = None
        self.start_of_line = True

    def next(self):
        if self.current_token == None:
            self.current_token = self.get_token()
        ret = self.current_token
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
            self.start_of_line = False
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
        elif is_digit(ch) or (ch == '.' and is_digit(self.stream.peak(1))):
            return self.parse_digit()
        elif ch == '\n':
            return self.parse_newline()
        elif is_separator(ch):
            return self.parse_separator()
        elif ch == '"' or ch == '\'':
            return self.parse_string_literal()
        else:
            err_msg = f"{self.stream.location_str} : Unrecognized token which start with {ch}"
            raise ValueError(err_msg)

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
            err_msg = f"{self.stream.location_str} : Should end with {quote} here"
            raise ValueError(err_msg)
        
        self.stream.next() # skip right quote
        return Token(TokenKind.StringLiteral, data)

    def parse_identifier(self):
        def is_valid(ch):
            return is_alpha(ch) or is_digit(ch) or ch == '_'
        
        data = ""
        while not self.stream.eof() and not is_separator(self.stream.peak()):
            ch = self.stream.next()
            data += ch
            if not is_valid(ch):
                err_msg = f"{self.stream.location_str} : Invalid identifier {data} with char {ch}"
                raise ValueError(err_msg)
        
        kind = TokenKind.Keyword if is_keyword(data) else TokenKind.Identifier
        return Token(kind, data)

    def parse_digit(self):
        def is_valid(ch):
            return is_digit(ch) or ch == '.'

        data = ""
        dot_cnt = 0
        while not self.stream.eof() and not is_separator(self.stream.peak()):
            ch = self.stream.next()
            data += ch
            if not is_valid(ch):
                err_msg = f"{self.stream.location_str} : Invalid number {data}"
                raise ValueError(err_msg)
            if ch == '.':
                if dot_cnt > 0:
                    err_msg = f"{self.stream.location_str} : Redundant decimal point {data}"
                    raise ValueError(err_msg)
                dot_cnt += 1
        kind = TokenKind.DecimalLiteral if dot_cnt > 0 else TokenKind.IntegerLiteral
        return Token(kind, data)

    def parse_separator(self):
        ch = self.stream.next()
        tok = Token(TokenKind.Separator, ch)
        if is_terminator(ch):
            tok.kind = TokenKind.Terminator
            tok.data = ""
        return tok

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
        
        err_msg = f"{self.stream.location_str} : Should end with ''' here"
        raise ValueError(err_msg)

    @property
    def location_str(self):
        return f"@(line: {self.stream.line}, col: {self.stream.col})"
