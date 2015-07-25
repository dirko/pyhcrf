import unittest

from numpy.testing import assert_array_almost_equal, assert_array_equal
import numpy as np
from scipy.sparse import csr_matrix
from pyhcrf import Hcrf
from pyhcrf.hcrf import forward_backward
from pyhcrf.hcrf import log_likelihood

TEST_PRECISION = 3


class TestHcrf(unittest.TestCase):
    def test_train_regression(self):
        # Lets add a test just to get everything working so we can refactor.
        X = [np.array([[1, 2], [5, 9], [7, 3]]),
             np.array([[6, -2], [3, 3]]),
             np.array([[1, -1]]),
             np.array([[1, 1], [5, 3], [4, 2], [3, 3]])]
        y = [0, 1, 0, 1]
        model = Hcrf(3)
        model.fit(X, y)
        actual = model.predict(X)

        expected = [0, 1, 0, 0]
        self.assertEqual(actual, expected)

    def test_train_regression_sparse(self):
        # Lets add a test just to get everything working so we can refactor.
        X = [csr_matrix((np.ones(3), np.array([1, 4, 7]), np.array(range(4))), shape=(3, 10)),
             csr_matrix((np.ones(3), np.array([3, 8, 7]), np.array(range(4))), shape=(3, 10)),
             csr_matrix((np.ones(1), np.array([3]), np.array(range(2))), shape=(1, 10)),
             csr_matrix((np.ones(4), np.array([1, 4, 7, 9]), np.array(range(5))), shape=(4, 10))]
        y = [0, 1, 0, 1]
        model = Hcrf(5, 1.0)
        model.fit(X, y)
        actual = model.predict(X)

        expected = [0, 0, 0, 1]
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
        A[1, 1, 0] = 1 * np.exp(1) * np.exp(8)
        A[1, 1, 1] = 1 * np.exp(0) * np.exp(7)

        A[2, 0, 0] = 1 * np.exp(1 + 8) * np.exp(3) * np.exp(-1)
        A[2, 0, 1] = 1 * np.exp(0 + 7) * np.exp(-2) * np.exp(1)
        A[2, 1, 0] = 1 * np.exp(1 + 8) * np.exp(2) * np.exp(1)
        A[2, 1, 1] = 1 * np.exp(0 + 7) * np.exp(1) * np.exp(-1)

        A[3, 0, 0] = 1 * np.exp(1 + 8) * np.exp(2 + 1) * np.exp(3) * np.exp(0)
        A[3, 0, 1] = 1 * np.exp(0 + 7) * np.exp(1 - 1) * np.exp(-2) * np.exp(-4)
        A[3, 1, 0] = 1 * np.exp(1 + 8 + 3 - 1) * np.exp(1 + 4) + 1 * np.exp(1 + 8 + 2 + 1) * np.exp(2 + 4)
        A[3, 1, 1] = 1 * np.exp(0 + 7 - 2 + 1) * np.exp(0 + 6) + 1 * np.exp(0 + 7 + 1 - 1) * np.exp(1 + 6)

        expected_forward_table = np.log(A)

        (forward_table,
         forward_transition_table,
         backward_table) = forward_backward(x,
                                            state_parameters,
                                            transition_parameters,
                                            transitions)
        #print np.log(A)
        print (forward_table)
        print backward_table
        np.testing.assert_array_almost_equal(forward_table, expected_forward_table)
        self.assertAlmostEqual(forward_table[-1, -1, 0], backward_table[0, 0, 0], places=5)
        self.assertAlmostEqual(forward_table[-1, -1, 1], backward_table[0, 0, 1], places=5)

    def test_gradient_small_transition(self):
        transitions = np.array([[0, 0, 0, 0],
                                [1, 0, 0, 1]])
        transition_parameters = np.array([1, 0], dtype='f64')
        x = np.array([[2, 3, -1]], dtype='f64').reshape(1, 3)
        state_parameters = np.array([[[-1, 2]],
                                     [[5, -6]],
                                     [[2, 13]]], dtype='f64')
        cy = 1

        delta = 5.0**-5
        for trans in range(len(transition_parameters)):
            tpd = np.zeros(transition_parameters.shape, dtype='f64')
            tpd[trans] = delta
            ll0, _, dtp0 = log_likelihood(x, cy, state_parameters, transition_parameters, transitions)
            ll1, _, dtp1 = log_likelihood(x, cy, state_parameters, transition_parameters + tpd, transitions)
            expected_der = (ll1 - ll0) / delta
            actual_der = dtp0[trans]
            print trans, '    ', expected_der, actual_der#, dsp0
            #print ll1, ll0, delta
            self.assertAlmostEqual(expected_der, actual_der, places=2)

    def test_gradient_small(self):
        transitions = np.array([[0, 0, 1, 0],
                                [1, 0, 1, 1],
                                [0, 1, 1, 2],
                                [1, 1, 1, 3],
                                [0, 1, 0, 4],
                                [1, 1, 0, 5]])
        transition_parameters = np.array([1, 0, 2, 1, 3, -2], dtype='f64')
        x = np.array([[2, 3, -1]], dtype='f64').reshape(1, 3)
        state_parameters = np.array([[[-1, 2],
                                      [3, -4]],
                                     [[5, -6],
                                      [7, 8]],
                                     [[-3, 6],
                                      [2, 13]]], dtype='f64')
        cy = 1

        delta = 5.0**-5
        for trans in range(len(transition_parameters)):
            tpd = np.zeros(transition_parameters.shape, dtype='f64')
            tpd[trans] = delta
            ll0, _, dtp0 = log_likelihood(x, cy, state_parameters, transition_parameters, transitions)
            ll1, _, dtp1 = log_likelihood(x, cy, state_parameters, transition_parameters + tpd, transitions)
            expected_der = (ll1 - ll0) / delta
            actual_der = dtp0[trans]
            print trans, '    ', expected_der, actual_der#, dsp0
            #print ll1, ll0, delta
            self.assertAlmostEqual(expected_der, actual_der, places=2)

    def test_gradient_large_state(self):
        transitions = np.array([[0, 0, 1, 0],
                                [1, 0, 1, 1],
                                [0, 1, 1, 2],
                                [1, 1, 1, 3],
                                [0, 1, 0, 4],
                                [1, 1, 0, 5]])
        transition_parameters = np.array([1, 0, 2, 1, 3, -2], dtype='f64')
        x = np.array([[2, 3, -1],
                      [1, 4, -2],
                      [5, 2, -3],
                      [-2, 5, 3]], dtype='f64')
        state_parameters = np.array([[[-1, 2],
                                      [3, -4]],
                                     [[5, -6],
                                      [7, 8]],
                                     [[-3, 6],
                                      [2, 13]]], dtype='f64')
        cy = 1
        delta = 5.0**-4

        K, S, W = state_parameters.shape
        for k in range(K):
            for s in range(S):
                for w in range(W):
                    spd = np.zeros(state_parameters.shape, dtype='f64')
                    spd[k, s, w] = delta
                    ll0, dsp0, _ = log_likelihood(x, cy, state_parameters, transition_parameters, transitions)
                    ll1, dsp1, _ = log_likelihood(x, cy, state_parameters + spd, transition_parameters, transitions)
                    expected_der = (ll1 - ll0) / delta
                    actual_der = dsp0[k, s, w]
                    print k, s, w, '    ', expected_der, actual_der#, dsp0
                    self.assertAlmostEqual(expected_der, actual_der, places=2)

    def test_gradient_large_transition(self):
        transitions = np.array([[0, 0, 1, 0],
                                [1, 0, 1, 1],
                                [0, 1, 1, 2],
                                [1, 1, 1, 3],
                                [0, 1, 0, 4],
                                [1, 1, 0, 5]])
        transition_parameters = np.array([1, -5, 20, 1, 3, -2], dtype='f64')
        x = np.array([[-2, 3, -1],
                      [1, 4, -2],
                      [4, -4, 2],
                      [3, 5, 3]], dtype='f64')
        state_parameters = np.array([[[-1, 2],
                                      [3, -4]],
                                     [[5, -6],
                                      [7, 8]],
                                     [[-3, 6],
                                      [2, 13]]], dtype='f64')
        cy = 1
        delta = 5.0**-5
        for trans in range(len(transition_parameters)):
            tpd = np.zeros(transition_parameters.shape, dtype='f64')
            tpd[trans] = delta
            ll0, _, dtp0 = log_likelihood(x, cy, state_parameters, transition_parameters, transitions)
            ll1, _, dtp1 = log_likelihood(x, cy, state_parameters, transition_parameters + tpd, transitions)
            expected_der = (ll1 - ll0) / delta
            actual_der = dtp0[trans]
            print trans, '    ', expected_der, actual_der#, dsp0
            #print ll1, ll0, delta
            #all_ders.append(abs(expected_der - actual_der) < 10.0**-2)
            self.assertAlmostEqual(expected_der, actual_der, places=2)
