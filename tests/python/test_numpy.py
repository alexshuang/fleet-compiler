import logging
import unittest
from utils import run_e2e_test


class TestNumpy(unittest.TestCase):
    def testNumpy(self):
        data = 'import numpy;assert(numpy.power(3, 3) == 27)'
        run_e2e_test(data)

    def testNp(self):
        data = 'import numpy as np;a = np.power(3, 3);assert(a == 27)'
        run_e2e_test(data)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    unittest.main()
