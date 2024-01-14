import logging
import unittest
from utils import run_e2e_test


class TestExpression(unittest.TestCase):
    def testExpression(self):
        data = 'a = 1 * 7 + (2 - 1 + 8**2);assert(a == 72)'
        run_e2e_test(data)

    def testExpression1(self):
        data = 'a = (2 - 1 + 8**2);assert(a == 65)'
        run_e2e_test(data)

    def testExpression2(self):
        data = 'a = 2 * 8**2;assert(a == 128)'
        run_e2e_test(data)

    def testExpression3(self):
        data = 'def foo(a):\n  return a ** 3\nassert(2 + 3 * foo(3) == 83)'
        run_e2e_test(data)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    unittest.main()
