# ===- parser.py -------------------------------------------------------------
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
from lexer import *

keywords = ['def', 'if', 'while', 'for', 'with']

class TokenKind(Enum):
    Keyword = 1
    Identifier = 2
    Separator = 3
    StringLiteral = 4
    IntegerLiteral = 5
    DecimalLiteral = 6
    NoneLiteral = 7
    EOF = 8


class Token:
    def __init__(self, kind=TokenKind.EOF, data="") -> None:
        self.kind = kind
        self.data = data
    

def is_alpha(ch):
    return (ch >= 'a' and ch <= 'z') or (ch >= 'A' and ch <= 'Z')
def is_digit(ch):
    return ch >= '0' and ch <= '9'
def is_separator(ch):
    return ch == ' ' or ch == '(' or ch == ')' or ch == '#' or ch == ':' or \
        ch == '[' or ch == ']' or ch == '=' or ch == ',' or ch == '!' or ch == '\n'


class Tokenizer:
    def __init__(self, data="") -> None:
        self.stream = CharStream(data)
        self.current_token = None

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

        self.skip_white_space()
        ch = self.stream.peak()

        # parse token 
        if is_alpha(ch):
            return self.parse_identifier()
        elif is_digit(ch) or (ch == '.' and is_digit(self.stream.peak(1))):
            return self.parse_digit()
        elif is_separator(ch):
            return self.parse_separator()
        elif ch == '#':
            self.skip_single_comment()
            return self.get_token()
        elif ch == '\'' and self.stream.peak(1) == '\'' and self.stream.peak(2) == '\'':
            self.skip_comments()
            return self.get_token()
        elif ch == '"' or ch == '\'':
            return self.parse_string_literal()
        else:
            err_msg = f"{self.stream.location_str} : Unrecognized token which start with {ch}"
            print(err_msg)
            raise ValueError(err_msg)

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
        
        kind = TokenKind.Keyword if data in keywords else TokenKind.Identifier
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
        return Token(TokenKind.Separator, self.stream.next())

    def skip_white_space(self):
        while self.stream.peak() == ' ':
            self.stream.next()

    def skip_single_comment(self):
        self.stream.next() # skip '#'
        while not self.stream.eof() and self.stream.peak() != '\n':
            self.stream.next()

    def skip_comments(self):
        for _ in range(3):
            self.stream.next() # skip "'''"
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
