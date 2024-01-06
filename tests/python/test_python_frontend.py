import logging
import unittest

import os
import io
import sys
import numpy as np
from fleet_compiler.frontend.python.parsing import Parser
from fleet_compiler.frontend.python.semantic import Pipeline, ReferenceResolvePass, ReplaceAliasOperationNamePass, BuiltinReferenceResolvePass
from fleet_compiler.frontend.python.runtime import Interpreter


def infer(data):
    output = io.StringIO()
    old_stdout = sys.stdout
    sys.stdout = output
    
    parser = Parser(data)
    module = parser.parse_module()

    pipeline = Pipeline()
    pipeline.add(ReferenceResolvePass())
    pipeline.add(ReplaceAliasOperationNamePass())
    pipeline.add(BuiltinReferenceResolvePass())
    pipeline.run(module, False)

    interpreter = Interpreter()
    interpreter.visit(module)

    sys.stdout = old_stdout
    return output


class TestPythonFrontend(unittest.TestCase):
    def testIndent(self):
        data = '''


        a = 1
        assert(a == 1)


        '''
        print(infer(data))

    def testExpression(self):
        data = '''
        a = 1 * 7 + (2 - 1 + 8**2)
        assert(a == 72)
        '''
        print(infer(data))
    
    # def testNumpyExpression(self):
    #     data = '''
    #     import numpy as np
        
    #     np.random.seed(42)
    #     a = np.random.randn(3, 4)
    #     b = np.random.randn(4, 5)
    #     c = np.random.randn(5)
    #     print(a @ b + c)
    #     '''
    #     print(infer(data))


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    unittest.main()
