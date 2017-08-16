import warnings
from asl_data import SinglesData


def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = []
    guesses = []

    for word_index, (X, lengths) in test_set.get_all_Xlengths().items():
        best_guess_word = None
        best_guess_score = float("-inf")
        word_probabilities = {}
        for model_word, model in models.items():
            try:
                score = model.score(X, lengths)
                word_probabilities[model_word] = score
                if best_guess_score < score:
                    best_guess_score = score
                    best_guess_word = model_word
            except:
                print("error scoring for model word {} with word index {}".format(model_word, word_index))

        probabilities.append(word_probabilities)
        guesses.insert(word_index, best_guess_word)
    return (probabilities, guesses)

