import logging
import unittest

from utils import *
from fleet_compiler.frontend.error import SyntaxErrorCode as code


class TestSyntax(unittest.TestCase):
    def testIndent1(self):
        data = '\n\na = 1\n'
        run_test(data)

    def testIndent2(self):
        data = '\n\n a = 1'
        run_test(data, code=code.Indentation)

    def testIndent3(self):
        data = '\n \na = 1'
        run_test(data, code=code.Indentation)

    def testIndent4(self):
        data = 'def foo():\n   print("hello world")\nfoo()'
        run_test(data)

    def testIndent5(self):
        data = 'def foo():\n  print("hello world")\n    \nfoo()'
        run_test(data, code=code.Indentation)

    def testDecimal1(self):
        data = 'a = -3.7;b=-2\nassert(a == -3.7)\nassert(b == -2)'
        run_e2e_test(data)

    def testDecimal2(self):
        data = 'a = -3.7.2'
        run_test(data, code=code.DecimalLiteral)

    def testStr(self):
        data = 'a = "hello comments\' b = 3\n'
        run_test(data, code=code.StringLiteral)

    def testComments(self):
        data = "''' hello comments ''"
        run_test(data, code=code.Comment)

    def testComments1(self):
        data = "''' hello comments '"
        run_test(data, code=code.Comment)

    def testComments2(self):
        data = "''' hello comments '''"
        run_test(data)

    def testReturnInSubBlock(self):
        data = 'def foo():\n  def bar():\n    return 20\n  return bar()\nassert(foo() == 20)'
        run_e2e_test(data)

    def testIf(self):
        data = 'x = 1\nif x <= 1:\n  y = 10\nelse:\n  y = 20\nassert(y == 10)'
        run_e2e_test(data)

    def testIf2(self):
        data = 'def fib(n):\n if n <= 1:\n  return n\n else:\n  return fib(n - 1) + fib(n - 2)\nassert(fib(6) == 8)'
        run_e2e_test(data)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    unittest.main()
