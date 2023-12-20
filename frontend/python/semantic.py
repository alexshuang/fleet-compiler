# ===- semantic.py -------------------------------------------------------------
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
# Semantic analysis:
#   Reference resolution: find the target function for the function call.
# 
# ===---------------------------------------------------------------------------

from frontend.python.syntax import FunctionCall
from syntax import *


class RefDumper(AstDumper):
    def visitFunctionCall(self, node: FunctionCall):
        ref_str = "(resolved)" if node.sym else "(not resolved)"
        print(self.prefix + f"Function Call {node.name}, args: {node.args}  {ref_str}")