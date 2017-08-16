import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences
import itertools


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states, X=None, lengths=None):
        if X is None:
            X = self.X
        if lengths is None:
            lengths = self.lengths
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(X, lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Bayesian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        best_model = None
        least_bic = float("inf")

        for n_of_components in range(self.min_n_components, self.max_n_components):
            model = self.base_model(n_of_components, self.X, self.lengths)
            if model is None:
                if self.verbose:
                    print("model failure on {} with {} states".format(self.this_word, n_of_components))
                continue
            try:
                score = model.score(self.X, self.lengths)
            except:
                if self.verbose:
                    print("score failure on {} with {} states".format(self.this_word, n_of_components))
                continue

            number_of_parameters = n_of_components ** 2.0 + 2.0 * len(self.X[0]) * n_of_components - 1.0

            logL = score
            p = number_of_parameters
            N = len(self.X)
            # BIC = −2 log L + p log N
            c_bic = (-2.0) * logL + p * np.log(N)
            if least_bic > c_bic:
                least_bic = c_bic
                best_model = model

        return best_model

class SelectorDIC(ModelSelector):

    def average_score_of_other_words(self, model, this_word, n_of_components):
        # get all words without this word
        total_score = 0.0
        count = 0
        for key, (X, lengths) in self.hwords.items():
            if key == this_word:
                continue

            try:
                score = model.score(X, lengths)
                total_score += score
                count += 1
            except:
                if self.verbose:
                    print("other word score failure on {} with {} states".format(key, n_of_components))
                continue

        return total_score / float(count)

    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        best_DIC = float("-inf")
        best_model = None

        for n_of_components in range(self.min_n_components, self.max_n_components):
            model = self.base_model(n_of_components, self.X, self.lengths)
            if model is None:
                if self.verbose:
                    print("model failure on {} with {} states".format(self.this_word, n_of_components))
                continue
            try:
                score = model.score(self.X, self.lengths)
            except:
                if self.verbose:
                    print("model failure on {} with {} states".format(self.this_word, n_of_components))
                continue

            other_words_average_score = self.average_score_of_other_words(model,
                                                                          self.this_word,
                                                                          n_of_components)

            #DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
            #DIC = current_logL - average_of_other_logLs
            c_dic = score - other_words_average_score

            if c_dic > best_DIC:
                best_DIC = c_dic
                best_model = model

        return best_model


class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    def select(self):
        """
        Implemented the cross-validation method to select the model with 
        the best average score of three different fold data sets while
        varying the number of states in the HMM
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        n_of_splits = min(len(self.sequences), 3)
        split_method = KFold(n_splits=n_of_splits)
        max_average_score = float("-inf")
        best_state_count = self.min_n_components

        for n_of_components in range(self.min_n_components, self.max_n_components):
            current_scores = []
            for cv_training_idx, cv_testing_idx in split_method.split(self.sequences):
                training_X, training_lengths = combine_sequences(cv_training_idx, self.sequences)
                testing_X, testing_lengths = combine_sequences(cv_testing_idx, self.sequences)
                model = self.base_model(n_of_components, training_X, training_lengths)
                if model is None:
                    if self.verbose:
                        print("model failure on {} with {} states".format(self.this_word, n_of_components))
                    continue
                try:
                    score = model.score(testing_X, testing_lengths)
                except:
                    if self.verbose:
                        print("score failure on {} with {} states".format(self.this_word, n_of_components))
                    continue

                current_scores.append(score)

            c_average_score = float("-inf")
            if current_scores:
                # get the average, if we have scores with the current number of
                # states
                c_average_score = np.average(current_scores)

            # pick the max score based on current number of state average score
            if max_average_score < c_average_score:
                max_average_score = c_average_score
                best_state_count = n_of_components

        return self.base_model(best_state_count, self.X, self.lengths)

