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
    def __init__(self,
                 num_states=2,
                 l2_regularization=0.0,
                 transitions=None,
                 state_parameter_noise=0.001,
                 transition_parameter_noise=0.001):
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
        if self.state_parameters is None:
            self.state_parameters = numpy.random.standard_normal((num_features,
                                                                  self.num_states + 2,
                                                                  num_classes)) * self.state_parameter_noise
        if self.transition_parameters is None:
            self.transition_parameters = numpy.random.standard_normal((num_transitions)) * self.transition_parameter_noise

        initial_parameter_vector = self._stack_parameters(self.state_parameters, self.transition_parameters)

        def objective_function(parameter_vector):
            ll = 0.0
            gradient = numpy.zeros_like(parameter_vector)
            state_parameters, transition_parameters = self._unstack_parameters(parameter_vector)
            for x, ty in zip(X, y):
                y_index = classes.index(y)
                dll, dgradient_state, dgradient_transition = log_likelihood(x,
                                                                            y_index,
                                                                            state_parameters,
                                                                            transition_parameters,
                                                                            self.transitions)
                dgradient = self._stack_parameters(dgradient_state, dgradient_transition)
                ll += dll
                gradient += dgradient
            return -ll, -gradient

        self._optimizer_result = fmin_l_bfgs_b(objective_function, initial_parameter_vector, **optimizer_kwargs)

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
        return [self.classes[prediction.argmax()] for prediction in self.predict_proba(X)]

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
            forward_table, _, _ = forward_backward(x,
                                                   self.state_parameters,
                                                   self.transition_parameters,
                                                   self.transitions)
            y.append(numpy.exp(forward_table[-1, -1, :]))
        return numpy.array(y)

    @staticmethod
    def _create_default_transitions(num_classes, num_states):
        num_transitions = num_classes * ((num_states * 2) + 2)
        transitions = numpy.zeros((num_transitions, 4))
        counter = 0
        for transition in range(num_states + 1):
            for c in range(num_classes):
                transitions[counter, 0] = c
                transitions[counter, 1] = transition
                transitions[counter, 2] = transition + 1
                transitions[counter, 3] = transition
                counter += 1
        return transitions

    @staticmethod
    def _stack_parameters(state_parameters, transition_parameters):
        return numpy.concatenate(state_parameters.flatten(), transition_parameters)

    def _unstack_parameters(self, parameter_vector):
        state_parameter_shape = self.state_parameters.shape
        num_state_parameters = numpy.prod(state_parameter_shape)
        return parameter_vector[:num_state_parameters].reshape(state_parameter_shape), parameter_vector[num_state_parameters:]


def forward_backward(x, state_parameters, transition_parameters, transitions):
    x_dot_parameters = numpy.tensordot(x, state_parameters, 1)
    print x_dot_parameters

    n_time_steps, n_states, n_classes = x_dot_parameters.shape
    n_transitions, _ = transitions.shape

    # Add extra 1 time steps, one for start state and one for end
    forward_table = numpy.full((n_time_steps + 1, n_states, n_classes), fill_value=-numpy.inf, dtype='f64')
    forward_transition_table = numpy.full((n_time_steps + 2, n_states, n_states, n_classes), fill_value=-numpy.inf, dtype='f64')
    forward_table[0, 0, :] = 0.0

    for t in range(1, n_time_steps + 1):
        for transition in range(n_transitions):
            class_number = transitions[transition, 0]
            s0 = transitions[transition, 1]
            s1 = transitions[transition, 2]
            edge_potential = forward_table[t - 1, s0, class_number] + transition_parameters[transition]
            forward_table[t, s1, class_number] = numpy.logaddexp(forward_table[t, s1, class_number],
                                                                 edge_potential + x_dot_parameters[t - 1, s1, class_number])
            forward_transition_table[t, s0, s1, class_number] = numpy.logaddexp(forward_transition_table[t, s0, s1, class_number],
                                                                                edge_potential)

    backward_table = numpy.full((n_time_steps + 1, n_states, n_classes), fill_value=-numpy.inf, dtype='f64')
    backward_table[-1, -1, :] = 0.0

    for t in range(n_time_steps, 0, -1):
        for transition in range(n_transitions):
            class_number = transitions[transition, 0]
            s0 = transitions[transition, 1]
            s1 = transitions[transition, 2]
            backward_table[t - 1, s0, class_number] = numpy.logaddexp(backward_table[t - 1, s0, class_number],
                                                                      backward_table[t, s1, class_number]
                                                                      + x_dot_parameters[t - 1, s1, class_number]
                                                                      + transition_parameters[transition])

    return forward_table, forward_transition_table, backward_table


def log_likelihood(x, cy, state_parameters, transition_parameters, transitions):
    forward_table, forward_transition_table, backward_table = forward_backward(x,
                                                                               state_parameters,
                                                                               transition_parameters,
                                                                               transitions)
    n_time_steps = forward_table.shape[0] - 1
    n_features, n_states, n_classes = state_parameters.shape
    n_transitions, _ = transitions.shape
    dstate_parameters = numpy.zeros_like(state_parameters, dtype='f64')
    dtransition_parameters = numpy.zeros_like(transition_parameters, dtype='f64')

    class_Z = {c: forward_table[-1, -1, c] for c in range(n_classes)}
    Z = -numpy.inf
    for c in range(n_classes):
        Z = numpy.logaddexp(Z, forward_table[-1, -1, c])

    for t in range(1, n_time_steps + 1):
        for state in range(n_states):
            for c in range(n_classes):
                alphabeta = forward_table[t, state, c] + backward_table[t, state, c]
                #print t, state, c, alphabeta, (numpy.exp(alphabeta - class_Z[c]) -
                #                                       numpy.exp(alphabeta - Z) * x[t - 1, :]), numpy.exp(alphabeta - Z) * x[t - 1, :]
                #print dstate_parameters
                if c == cy:
                    dstate_parameters[:, state, c] += (numpy.exp(alphabeta - class_Z[c]) -
                                                       numpy.exp(alphabeta - Z) * x[t - 1, :])
                else:
                    dstate_parameters[:, state, c] -= numpy.exp(alphabeta - Z) * x[t - 1, :]

    for t in range(1, n_time_steps + 1):
        for transition in range(n_transitions):
            c = transitions[transition, 0]
            s0 = transitions[transition, 1]
            s1 = transitions[transition, 2]
            alphabeta = forward_transition_table[t, s0, s1, c] + backward_table[t, s1, c]
            if c == cy:
                dtransition_parameters[transition] += (numpy.exp(alphabeta - class_Z[c]) - numpy.exp(alphabeta - Z))
            else:
                dtransition_parameters[transition] -= numpy.exp(alphabeta - Z)

    return class_Z[cy] - Z, dstate_parameters, dtransition_parameters
