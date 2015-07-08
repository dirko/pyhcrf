import unittest

from numpy.testing import assert_array_almost_equal, assert_array_equal
import numpy as np
from numpy import random
from pyhcrf import Hcrf

TEST_PRECISION = 3


class TestHcrf(unittest.TestCase):
    def test_train_regression_a(self):
        # Lets add a test just to get everything working so we can refactor.
        X = [[[1], [5], [7]], [[6], [3]], [[1]], [[1], [5], [4]]]
        y = [0, 1, 0, 1]
        model = Hcrf(3)
        model.fit(X, y)
        actual = model.predict(X)

        expected = [0, 1, 0, 0]
        self.assertEqual(actual, expected)

    def test_train_regression_b(self):
        # Lets add a test just to get everything working so we can refactor.
        X = [[[1], [5], [7]], [[6], [3]], [[1]], [[1], [5], [4]]]
        y = [0, 1, 0, 1]
        model = Hcrf(5, 1.0)
        model.fit(X, y)
        actual = model.predict(X)

        expected = [0, 1, 0, 1]
        self.assertEqual(actual, expected)
