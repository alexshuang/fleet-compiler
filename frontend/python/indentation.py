# ===- indentation.py -------------------------------------------------------------
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

class Indentation:
    def __init__(self, parent=None) -> None:
        self.data = None
        self.parent = parent # parent scope

    def match(self, data: str):
        if self.data:
            if self.data != data:
                return False
        else:
            self.data = data
        return True
    
    def match_parents(self, data: str):
        if self.parent.data == data:
            return True
        else:
            return self.parent.match_parents(data) if self.parent else None
