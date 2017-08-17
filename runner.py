import numpy as np
import pandas as pd
from asl_data import AslDb

asl = AslDb()

features_ground = ['grnd-rx','grnd-ry','grnd-lx','grnd-ly']
asl.df['grnd-ry'] = asl.df['right-y'] - asl.df['nose-y']
asl.df['grnd-rx'] = asl.df['right-x'] - asl.df['nose-x']
asl.df['grnd-ly'] = asl.df['left-y'] - asl.df['nose-y']
asl.df['grnd-lx'] = asl.df['left-x'] - asl.df['nose-x']

#words_to_train = ['FISH']
words_to_train = ['FISH', 'BOOK', 'VEGETABLE', 'FUTURE', 'JOHN']
import timeit

from my_model_selectors import SelectorCV, SelectorBIC, SelectorDIC

training = asl.build_training(features_ground)
sequences = training.get_all_sequences()
Xlengths = training.get_all_Xlengths()

for word in words_to_train:
    start = timeit.default_timer()
    #model = SelectorCV(sequences, Xlengths, word, 
    #                min_n_components=2, max_n_components=15, random_state = 14).select()
#    model = SelectorBIC(sequences, Xlengths, word, 
#                    min_n_components=2, max_n_components=15, random_state = 14).select()
    model = SelectorDIC(sequences, Xlengths, word, 
                    min_n_components=2, max_n_components=15, random_state = 14).select()
    end = timeit.default_timer()-start
    if model is not None:
        print("Training complete for {} with {} states with time {} seconds".format(word, model.n_components, end))
    else:
        print("Training failed for {}".format(word))
print("end")
