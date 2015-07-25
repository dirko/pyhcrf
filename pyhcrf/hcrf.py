# File: hcrf.py
# Author: Dirko Coetsee
# Date: 13 Augustus 2013
# Updated: 22 July 2015 - almost complete re-write to use sklearn interface.

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
                 random_seed=0,
                 verbosity=0):
        """
        Initialize new HCRF object with hidden units with cardinality `num_states`.
        """
        self.lamb = l2_regularization
        assert(num_states > 0)
        self.num_states = num_states
        self.classes_ = None
        self.state_parameters = None
        self.transition_parameters = None
        self.transitions = transitions
        self.state_parameter_noise = state_parameter_noise
        self.transition_parameter_noise = transition_parameter_noise
        self.optimizer_kwargs = optimizer_kwargs if optimizer_kwargs else {}
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
        function_evaluations = [1]

        def objective_function(parameter_vector):
            ll = 0.0
            gradient = numpy.zeros_like(parameter_vector)
            state_parameters, transition_parameters = self._unstack_parameters(parameter_vector)
            for x, ty in zip(X, y):
                y_index = classes.index(ty)
                dll, dgradient_state, dgradient_transition = log_likelihood(x,
                                                                            y_index,
                                                                            state_parameters,
                                                                            transition_parameters,
                                                                            self.transitions)
                dgradient = self._stack_parameters(dgradient_state, dgradient_transition)
                ll += dll
                gradient += dgradient
            function_evaluations[0] += 1
            if self._verbosity > 0 and function_evaluations[0] % self._verbosity == 0:
                print '{:10} {:10.2f} {:10.2f}'.format(function_evaluations[0], ll, sum(abs(gradient)))
            return -ll, -gradient

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
            y.append(numpy.exp(forward_table[-1, -1, :]))
        return numpy.array(y)

    @staticmethod
    def _create_default_transitions(num_classes, num_states):
        # 0  o>
        # 1  o>
        # 2  o>
        num_transitions = num_classes * ((num_states * 2) - 1)
        transitions = numpy.zeros((num_transitions, 3), dtype='int64')
        counter = 0
        for c in range(num_classes):  # The zeroth state
            transitions[counter, 0] = c
            transitions[counter, 1] = 0
            transitions[counter, 2] = 0
            counter += 1
        for state in range(0, num_states - 1):  # Subsequent states
            for c in range(num_classes):
                transitions[counter, 0] = c  # To the next state
                transitions[counter, 1] = state
                transitions[counter, 2] = state + 1
                counter += 1
                transitions[counter, 0] = c  # Stays in same state
                transitions[counter, 1] = state + 1
                transitions[counter, 2] = state + 1
                counter += 1
        return transitions

    @staticmethod
    def _stack_parameters(state_parameters, transition_parameters):
        return numpy.concatenate((state_parameters.flatten(), transition_parameters))

    def _unstack_parameters(self, parameter_vector):
        state_parameter_shape = self.state_parameters.shape
        num_state_parameters = numpy.prod(state_parameter_shape)
        return parameter_vector[:num_state_parameters].reshape(state_parameter_shape), parameter_vector[num_state_parameters:]
