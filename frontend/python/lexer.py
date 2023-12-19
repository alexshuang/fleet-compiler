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
    