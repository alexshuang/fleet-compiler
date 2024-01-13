import logging
import unittest
import utils


def run_test(data):
    utils.run_test(data, utils.get_basic_pipeline(), utils.get_interpreter())


class TestNumpy(unittest.TestCase):
    def testNumpy(self):
        data = 'import numpy;assert(numpy.power(3, 3) == 27)'
        run_test(data)

    def testNp(self):
        data = 'import numpy as np;a = np.power(3, 3);assert(a == 27)'
        run_test(data)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    unittest.main()
