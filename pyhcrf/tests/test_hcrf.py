import unittest

from numpy.testing import assert_array_almost_equal, assert_array_equal
import numpy as np
from numpy import random
from pyhcrf import Hcrf
from pyhcrf.hcrf import forward_backward

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

    def test_forward_backward(self):
        transitions = np.array([[0, 0, 1, 0],
                                [1, 0, 1, 1],
                                [0, 1, 1, 2],
                                [1, 1, 1, 3],
                                [0, 1, 0, 4],
                                [1, 1, 0, 5]])
        transition_parameters = np.array([1, 0, 2, 1, 3, -2])
        x = np.array([[2, 3],
                      [1, 0],
                      [0, 2]])
        state_parameters = np.array([[[-1, 1],
                                      [1, -1]],
                                     [[0, -2],
                                      [2, 3]]])
        print 'X.L'
        print np.dot(x, state_parameters)
        A = np.zeros((4, 2, 2), dtype='f64')
        # T  S  W
        # 5, 2, 2
        A[0, 0, 0] = 1
        A[0, 0, 1] = 1
        A[0, 1, 0] = 0
        A[0, 1, 1] = 0

        A[1, 0, 0] = 0
        A[1, 0, 1] = 0
        A[1, 1, 0] = 1 * np.exp(1) * np.exp(6)
        A[1, 1, 1] = 1 * np.exp(0) * np.exp(5)

        A[2, 0, 0] = 1 * np.exp(1 + 6) * np.exp(3) * np.exp(-1)
        A[2, 0, 1] = 1 * np.exp(0 + 5) * np.exp(-2) * np.exp(1)
        A[2, 1, 0] = 1 * np.exp(1 + 6) * np.exp(2) * np.exp(0)
        A[2, 1, 1] = 1 * np.exp(0 + 5) * np.exp(1) * np.exp(-2)

        A[3, 0, 0] = 1 * np.exp(1 + 6) * np.exp(2 + 0) * np.exp(3) * np.exp(2)
        A[3, 0, 1] = 1 * np.exp(0 + 5) * np.exp(1 - 2) * np.exp(-2) * np.exp(-2)
        A[3, 1, 0] = 1 * np.exp(1 + 6 + 3 - 1) * np.exp(1 + 4) + 1 * np.exp(1 + 6 + 2 + 0) * np.exp(2 + 4)
        A[3, 1, 1] = 1 * np.exp(0 + 5 - 2 + 1) * np.exp(0 + 6) + 1 * np.exp(0 + 5 + 1 - 2) * np.exp(1 + 6)

        expected_forward_table = np.log(A)

        forward_table, forward_transition_table, backward_table = forward_backward(x,
                                                                                   state_parameters,
                                                                                   transition_parameters,
                                                                                   transitions)
        #print np.log(A)
        print (forward_table)
        print backward_table
        np.testing.assert_array_almost_equal(forward_table, expected_forward_table)
        self.assertAlmostEqual(forward_table[-1, -1, 0], backward_table[0, 0, 0], places=5)
        self.assertAlmostEqual(forward_table[-1, -1, 1], backward_table[0, 0, 1], places=5)
