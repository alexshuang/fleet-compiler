import logging
import unittest
import utils


def run_test(data):
    utils.run_test(data, utils.get_basic_pipeline(), utils.get_interpreter())


class TestExpression(unittest.TestCase):
    def testExpression(self):
        data = 'a = 1 * 7 + (2 - 1 + 8**2);assert(a == 72)'
        run_test(data)

    def testExpression1(self):
        data = 'a = (2 - 1 + 8**2);assert(a == 65)'
        run_test(data)

    def testExpression2(self):
        data = 'a = 2 * 8**2;assert(a == 128)'
        run_test(data)



if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    unittest.main()
