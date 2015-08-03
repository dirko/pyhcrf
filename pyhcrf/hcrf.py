# File: hcrf.py
# Author: Dirko Coetsee
# License: GPL
# (Contact me if this is a problem.)
# Date: 13 Augustus 2013
# Updated: 22 July 2015 - almost complete re-write to use sklearn-type interface.
#           3 Aug 2015 - Done with new interface.

import numpy
from scipy.optimize.lbfgsb import fmin_l_bfgs_b
from algorithms import forward_backward, log_likelihood


class Hcrf(object):
    """
    The HCRF model.

    Includes methods for training using LM-BFGS, scoring, and testing, and
    helper methods for loading and saving parameter values to and from file.
    """
    def __init__(self,
                 num_states=2,
                 l2_regularization=1.0,
                 transitions=None,
                 state_parameter_noise=0.001,
                 transition_parameter_noise=0.001,
                 optimizer_kwargs=None,
                 sgd_stepsize=None,
                 sgd_verbosity=None,
                 random_seed=0,
                 verbosity=0):
        """
        Initialize new HCRF object with hidden units with cardinality `num_states`.
        """
        self.l2_regularization = l2_regularization
        assert(num_states > 0)
        self.num_states = num_states
        self.classes_ = None
        self.state_parameters = None
        self.transition_parameters = None
        self.transitions = transitions
        self.state_parameter_noise = state_parameter_noise
        self.transition_parameter_noise = transition_parameter_noise
        self.optimizer_kwargs = optimizer_kwargs if optimizer_kwargs else {}
        self._sgd_stepsize = sgd_stepsize
        self._sgd_verbosity = sgd_verbosity
        self._random_seed = random_seed
        self._verbosity = verbosity

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
        classes = list(set(y))
        num_classes = len(classes)
        self.classes_ = classes
        if self.transitions is None:
            self.transitions = self._create_default_transitions(num_classes, self.num_states)

        # Initialise the parameters
        _, num_features = X[0].shape
        num_transitions, _ = self.transitions.shape
        numpy.random.seed(self._random_seed)
        if self.state_parameters is None:
            self.state_parameters = numpy.random.standard_normal((num_features,
                                                                  self.num_states,
                                                                  num_classes)) * self.state_parameter_noise
        if self.transition_parameters is None:
            self.transition_parameters = numpy.random.standard_normal((num_transitions)) * self.transition_parameter_noise

        initial_parameter_vector = self._stack_parameters(self.state_parameters, self.transition_parameters)
        function_evaluations = [0]

        def objective_function(parameter_vector, batch_start_index=0, batch_end_index=-1):
            ll = 0.0
            gradient = numpy.zeros_like(parameter_vector)
            state_parameters, transition_parameters = self._unstack_parameters(parameter_vector)
            for x, ty in zip(X, y)[batch_start_index: batch_end_index]:
                y_index = classes.index(ty)
                dll, dgradient_state, dgradient_transition = log_likelihood(x,
                                                                            y_index,
                                                                            state_parameters,
                                                                            transition_parameters,
                                                                            self.transitions)
                dgradient = self._stack_parameters(dgradient_state, dgradient_transition)
                ll += dll
                gradient += dgradient

            parameters_without_bias = numpy.array(parameter_vector)  # exclude the bias parameters from being regularized
            parameters_without_bias[0] = 0
            ll -= self.l2_regularization * numpy.dot(parameters_without_bias.T, parameters_without_bias)
            gradient = gradient.flatten() - 2.0 * self.l2_regularization * parameters_without_bias

            if batch_start_index == 0:
                function_evaluations[0] += 1
                if self._verbosity > 0 and function_evaluations[0] % self._verbosity == 0:
                    print '{:10} {:10.2f} {:10.2f}'.format(function_evaluations[0], ll, sum(abs(gradient)))
            return -ll, -gradient

        # If the stochastic gradient stepsize is defined, do 1 epoch of SGD to initialize the parameters.
        if self._sgd_stepsize:
            total_nll = 0.0
            for i in range(len(y)):
                nll, ngradient = objective_function(initial_parameter_vector, i, i + 1)
                total_nll += nll
                initial_parameter_vector -= ngradient * self._sgd_stepsize
                if self._sgd_verbosity > 0:
                    if i % self._sgd_verbosity == 0:
                        print '{:10} {:10.2f} {:10.2f}'.format(i, -total_nll / (i + 1) * len(y), sum(abs(ngradient)))

        self._optimizer_result = fmin_l_bfgs_b(objective_function, initial_parameter_vector, **self.optimizer_kwargs)
        self.state_parameters, self.transition_parameters = self._unstack_parameters(self._optimizer_result[0])
        return self

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
        return [self.classes_[prediction.argmax()] for prediction in self.predict_proba(X)]

    def predict_proba(self, X):
        """Probability estimates.

        The returned estimates for all classes are ordered by the
        label of classes.

        Parameters
        ----------
        X : List of ndarrays, one for each training example.

        Returns
        -------
        T : array-like, shape = [n_samples, n_classes]
            Returns the probability of the sample for each class in the model,
            where classes are ordered as they are in ``self.classes_``.
        """
        y = []
        for x in X:
            n_time_steps, n_features = x.shape
            _, n_states, n_classes = self.state_parameters.shape
            x_dot_parameters = x.dot(self.state_parameters.reshape(n_features, -1)).reshape((n_time_steps, n_states, n_classes))
            forward_table, _, _ = forward_backward(x_dot_parameters, self.state_parameters, self.transition_parameters, self.transitions)
            # TODO: normalize by subtracting log-sum to avoid overflow
            y.append(numpy.exp(forward_table[-1, -1, :]) / sum(numpy.exp(forward_table[-1, -1, :])))
        return numpy.array(y)

    @staticmethod
    def _create_default_transitions(num_classes, num_states):
        # 0    o>
        # 1    o>\\\
        # 2   /o>/||
        # 3  |/o>//
        # 4  \\o>/
        transitions = []
        for c in range(num_classes):  # The zeroth state
            transitions.append([c, 0, 0])
        for state in range(0, num_states - 1):  # Subsequent states
            for c in range(num_classes):
                transitions.append([c, state, state + 1])  # To the next state
                transitions.append([c, state + 1, state + 1])  # Stays in same state
                if state > 0:
                    transitions.append([c, 0, state + 1])  # From the start state
                if state < num_states - 1:
                    transitions.append([c, state + 1, num_states - 1])  # To the end state
        transitions = numpy.array(transitions, dtype='int64')
        return transitions

    @staticmethod
    def _stack_parameters(state_parameters, transition_parameters):
        return numpy.concatenate((state_parameters.flatten(), transition_parameters))

    def _unstack_parameters(self, parameter_vector):
        state_parameter_shape = self.state_parameters.shape
        num_state_parameters = numpy.prod(state_parameter_shape)
        return parameter_vector[:num_state_parameters].reshape(state_parameter_shape), parameter_vector[num_state_parameters:]
