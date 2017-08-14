import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


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

        # TODO implement model selection based on BIC scores
        raise NotImplementedError


class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection based on DIC scores
        raise NotImplementedError


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
        final_max_model = None

        for n_of_components in range(self.min_n_components, self.max_n_components):
            c_max_model = None
            current_scores = []
            c_max_score = float("-inf")
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
                if c_max_score < score:
                    c_max_model = model
                    c_max_score = score

            c_average_score = float("-inf")
            if current_scores:
                c_average_score = np.average(current_scores)

            if max_average_score < c_average_score:
                max_average_score = c_average_score
                final_max_model = c_max_model

        return final_max_model
