# File: hcrf.py
# Author: Dirko Coetsee
# Date: 13 Augustus 2013
# A script to train and test an HCRF for sparse input vectors.
#
# TODO: - Add support for more than one feature on each time step.
#       - Add feature weights.
#       - Change inference to use more efficient matrix routines.

import numpy
from random import random, seed
import sys
from scipy.optimize.lbfgsb import fmin_l_bfgs_b


class Hcrf(object):
    """
    The HCRF model.

    Includes methods for training using LM-BFGS, scoring, and testing, and
    helper methods for loading and saving parameter values to and from file.
    """
    def __init__(self, S, l2_regularization=0.0):
        """
        Initialize new HCRF object with hidden units with cardinality S.
        """
        self.lamb = l2_regularization

        self.S = S

        self.A = None
        self.B = None
        self.C = None
        self.D = None

    def fit(self, X, y):
        """Fit the model according to the given training data.

        Parameters
        ----------
        X : List of list of ints. Each list of ints represent a training example. Each int in that list
            is the index of a one-hot encoded feature.

        y : array-like, shape (n_samples,)
            Target vector relative to X.

        Returns
        -------
        self : object
            Returns self.
        """
        self.X = X
        self.Y = y

        F = max(max(max(eeex for eeex in eex) for eex in ex) for ex in X) + 1
        W = max(y) + 1
        self.W = W
        S = self.S

        # Initialize the parameters uniformly random on [0, 0.1]
        seed(2)
        self.param = numpy.array([random() * 0.1 for i in xrange(S * S + S * W * F)])
        for s in xrange(S):
            for ps in xrange(S):
                # Make it impossible for hidden units to remain in state 0.
                if ps == 0 and s == 0:
                    self.param[s * S + ps] = -numpy.inf
                # Hidden units can also not stay in last state S-1.
                if ps == S - 1 and s == S - 1:
                    self.param[s * S + ps] = -numpy.inf
                # Hidden states can only go to higher state or stay in the
                # same state.
                if s < ps:
                    self.param[s * S + ps] = -numpy.inf
        self.param_non_inf_indexes = [i for i in xrange(len(self.param)) if self.param[i] != -numpy.inf]

        # Train
        final_params = self.train_lmbfgs()
        self.param[self.param_non_inf_indexes] = final_params[0]

    def predict(self, X):
        """Predict the class for X.

        The predicted class for each sample in X is returned.

        Parameters
        ----------
        X : List of list of ints, one list of ints for each training example.

        Returns
        -------
        y : iterable of shape = [n_samples]
            The predicted classes.
        """
        y = []
        for x in X:
            pred = self.load_example(x)
            py = max(zip(pred, range(len(pred))), key=lambda element: element[0])[1]
            y.append(py)
        return y

    def reset_tables(self):
        T = self.T
        S = self.S
        W = self.W
        self.A = numpy.zeros((T, S, W))
        self.B = numpy.zeros((T, S, W))
        self.C = numpy.zeros((T, S, W))
        self.D = numpy.zeros((T, S, S, W))

    def reset_deriv(self):
        self.der = numpy.zeros(len(self.param))
        self.ll = 0.0

    def load_example(self, x):
        # Fill forward, backward, and combination tables for example x
        self.T = len(x)
        self.reset_tables()
        self.x = x
        self.fill_A()
        self.fill_B()
        self.fill_C()
        # Return the predicted distribution over classes
        return [sum(self.C[self.T - 1, :, y]) for y in xrange(self.W)]

    def get_deriv(self, x, y):
        # Return the log-likelihood and derivative of the parameters for
        # a training example with features x and label y.
        self.load_example(x)
        self.fill_D()
        D = self.D
        W = self.W
        S = self.S
        T = self.T
        C = self.C
        der = self.der
        ll = numpy.log(sum(C[T - 1, :, y]))
        self.ll += ll

        # Factors without output variable interaction
        for t in xrange(T):
            E_ef_norm = sum(C[t, :, y])
            for w in xrange(W):
                for s in xrange(S):

                    E_f = C[t, s, w]
                    E_ef = C[t, s, w]
                    if w != y:
                        E_ef = 0
                    f_list = self.get_fs(t, s, w)
                    E_ef_n = E_ef / E_ef_norm

                    for f in f_list:
                        der[f] += E_ef / E_ef_norm - E_f
        # Factors containing two output variables
        for t in xrange(T - 1):
            E_ef_norm = sum(sum(D[t, :, :, y]))
            for w in xrange(W):
                for s in xrange(S):
                    for ps in xrange(S):
                        E_f = D[t, s, ps, w]
                        E_ef = D[t, s, ps, w]
                        if w != y:
                            E_ef = 0
                        f_list = self.get_fss(t, s, ps, w)
                        E_ef_n = E_ef / E_ef_norm

                        for f in f_list:
                            der[f] += E_ef_n - E_f

        return ll, der

    def fill_A(self):
        # Fill the forward table.
        T = self.T
        S = self.S
        W = self.W
        A = self.A
        for w in xrange(W):
            A[0, 0, w] = numpy.exp(self.get_f_single(0, 0, w))

        for t in xrange(1, T):
            for w in xrange(W):
                for s in xrange(S):
                    for ps in xrange(S):
                        A[t, s, w] = A[t, s, w] + A[t - 1, ps, w] * numpy.exp(self.get_f(t, s, ps, w))
            norm = sum(sum(A[t, :, :]))
            A[t, :, :] /= norm

    def fill_B(self):
        # Fill the backward table.
        T = self.T
        S = self.S
        W = self.W
        B = self.B
        for w in xrange(W):
            B[T - 1, S - 1, w] = numpy.exp(self.get_f_single(T - 1, S - 1, w))
        for t in xrange(T - 2, -1, -1):
            for w in xrange(W):
                for s in xrange(S):
                    for ps in xrange(S):
                        B[t, ps, w] = B[t, ps, w] + B[t + 1, s, w] * numpy.exp(self.get_f(t + 1, s, ps, w))
            norm = sum(sum(B[t, :, :]))
            B[t, :, :] /= norm

    def fill_C(self):
        # Fill and normalize table with product of forward and backward tables.
        self.C = self.A * self.B
        for t in xrange(self.T):
            norm = sum(sum(self.C[t, :, :]))
            self.C[t, :, :] = self.C[t, :, :] / norm

    def fill_D(self):
        # Get probability table.
        T = self.T
        S = self.S
        W = self.W
        A = self.A
        B = self.B
        D = self.D
        for t in xrange(T - 1):
            for w in xrange(W):
                for s in xrange(S):
                    for ps in xrange(S):
                        D[t, s, ps, w] = A[t, ps, w] * B[t + 1, s, w] * numpy.exp(self.get_f(t + 1, s, ps, w))
            norm = sum(sum(sum(D[t, :, :, :])))
            D[t, :, :, :] /= norm

    def get_f(self, t, p, ps, w):
        # Fill cell in forward or backward table.
        f_list = self.get_fs(t, p, w)
        f_list += self.get_fss(t, p, ps, w)
        return sum(self.param[f] for f in f_list)

    def get_f_single(self, t, p, w):
        # Potential of non-interaction cell
        f_list = self.get_fs(t, p, w)
        return sum(self.param[f] for f in f_list)

    def get_fs(self, t, p, w):
        # Get list of parameters activated at time t, hidden variable p,
        # and output variable w.
        S = self.S
        W = self.W
        flist = []
        for f in self.x[t]:
            flist += [S * S + (p * W + w) + W * S * f]
        return flist

    def get_fss(self, t, p, ps, w):
        # Get list of parameters activated at time t, hidden variable p,
        # previous hidden variable ps, and output variable w.
        S = self.S
        return [p * S + ps]

    def train_lmbfgs(self):
        """
        Train the model by maximising posterior with LM-BFGS.

        The training data should have been set at this stage:
            >> h = hcrf(H, maxw, maxf)
            >> h.X = X
            >> h.Y = Y
            >> h.lamb = lamb
            >> final_params = h.train_lmbfgs()
        Return the final parameter vector.
        """
        initial = self.param[self.param_non_inf_indexes]
        fparam = fmin_l_bfgs_b(self.get_obj, initial)
        return fparam

    def get_obj(self, npar, *args):
        # Use get_deriv to find the objective and its derivative for training.
        self.param[self.param_non_inf_indexes] = npar
        self.reset_deriv()
        for x, y in zip(self.X, self.Y):
            self.get_deriv(x, y)
        der = self.der[self.param_non_inf_indexes]
        ll = -self.ll - (-sum(self.lamb * 0.5 * xx ** 2.0 for xx in npar))
        der = -der - (-npar * self.lamb)
        # Print the log-likelihood
        print self.ll
        print ll
        print der
        return ll, der

    def test(self):
        """
        Test the current input data on the current model.

        Prints a confusion matrix, the number of correctly labeled examples,
        the total number of examples, and accuracy to standard out.
        """
        total = 0
        cor = 0
        conf = numpy.zeros((self.W, self.W))
        for x, y in zip(self.X, self.Y):
            pred = self.load_example(x)
            py = max(zip(pred, range(len(pred))), key=lambda x: x[0])[1]
            if py == y:
                cor += 1
            conf[y, py] += 1
            total += 1
        print conf
        print cor, total, cor * 1.0 / total

    def label(self):
        """
        Predict the labels of current input data and print to standard out.
        """
        for x in X:
            pred = self.load_example(x)
            py = max(zip(pred, range(len(pred))), key=lambda x: x[0])[1]
            print py

    def load_params(self, filename):
        """
        Take parameters from file and set to current parameters.
        """
        f = open(filename, 'r')
        p = numpy.array([float(line.strip("\n")) for line in f])
        self.param[self.param_non_inf_indexes] = p

    def save_params(self, filename):
        """
        Take current parameters and write to file.
        """
        f = open(filename, 'w')
        for p in self.param[self.param_non_inf_indexes]:
            f.write(str(p) + "\n")
        f.close()


def load_data(filename):
    """
    Load training or testing data from file.

    Return input vectors X, labels Y, cardinality of y in Y, and cardinality
    of x in X.
    """
    f = open(filename, "r")
    X = []
    Y = []
    maxf = 0
    maxw = 0
    for line in f:
        toks = [int(tok) for tok in line.strip(" \n").split(" ")]
        y = toks[0]
        if int(y) > maxw:
            maxw = int(y)
        x = [[tok] for tok in toks[1:]]
        for t in toks[1:]:
            if t > maxf:
                maxf = t
        Y += [y]
        X += [x]
    return X, Y, maxf + 1, maxw + 1


if __name__ == "__main__":
    usage = """
    Train, test, or use an HCRF to label input data.

    usage: hcrf.py mode datafile paramfile H [lamb]

    mode:      Set the script mode to train, tst, or label.
    datafile:  File containing input datapoints.
               Format: lines consisting of
               label 1 [feat1 feat2 ... ] 2
               where label is a non-negative integer, 1 is the special start
               of datapoint feature, 2 is the special end of datapoint feature,
               and feat1, feat2 etc. are integers > 2 representing features
               activated at the first, second etc. time steps.
    paramfile: File to store/retrieve parameters.
    H:         Cardinality of hidden units. Must be >= 3.
    lamb:      l2 reguralization constant. Only applicable when mode is train.
    """
    datafile = sys.argv[2]
    paramfile = sys.argv[3]
    H = int(sys.argv[4])
    if sys.argv[1] == "train":
        lamb = float(sys.argv[5])
        X, Y, maxf, maxw = load_data(datafile)
        h = Hcrf(H, maxw, maxf)
        h.X = X
        h.Y = Y
        h.lamb = lamb
        final_params = h.train_lmbfgs()
        h.param[h.param_non_inf_indexes] = final_params[0]
        h.save_params(paramfile)

    if sys.argv[1] == "tst":
        X, Y, maxf, maxw = load_data(datafile)
        h = Hcrf(H, maxw, maxf)
        h.load_params(paramfile)
        h.X = X
        h.Y = Y
        h.test()

    if sys.argv[1] == "label":
        X, Y, maxf, maxw = load_data(datafile)
        h = Hcrf(H, maxw, maxf)
        h.load_params(paramfile)
        h.X = X
        h.label()
