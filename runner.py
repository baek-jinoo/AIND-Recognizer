import numpy as np
import pandas as pd
from asl_data import AslDb

asl = AslDb()

features_ground = ['grnd-rx','grnd-ry','grnd-lx','grnd-ly']
asl.df['grnd-ry'] = asl.df['right-y'] - asl.df['nose-y']
asl.df['grnd-rx'] = asl.df['right-x'] - asl.df['nose-x']
asl.df['grnd-ly'] = asl.df['left-y'] - asl.df['nose-y']
asl.df['grnd-lx'] = asl.df['left-x'] - asl.df['nose-x']

#df_means = asl.df.groupby('speaker').mean()
#df_std = asl.df.groupby('speaker').std()
#
#features_norm = ['norm-rx', 'norm-ry', 'norm-lx','norm-ly']
#features_norm_map = {'norm-rx': 'right-x', 'norm-ry': 'right-y', 'norm-lx': 'left-x','norm-ly': 'left-y'}
#
#def std(column, local_df, local_df_means, local_df_std):
#    means = local_df['speaker'].map(local_df_means[column])
#    standards = local_df['speaker'].map(local_df_std[column])
#    return (local_df[column] - means) / standards
#
#for feature_norm in features_norm:
#    asl.df[feature_norm] = std(features_norm_map[feature_norm], asl.df, df_means, df_std)
#
#features_polar = ['polar-rr', 'polar-rtheta', 'polar-lr', 'polar-ltheta']
#
#asl.df[features_polar[0]] = np.sqrt(asl.df['grnd-rx']**2 + asl.df['grnd-ry']**2)
#asl.df[features_polar[1]] = np.arctan2(asl.df['grnd-rx'], asl.df['grnd-ry'])
#asl.df[features_polar[2]] = np.sqrt(asl.df['grnd-lx']**2 + asl.df['grnd-ly']**2)
#asl.df[features_polar[3]] = np.arctan2(asl.df['grnd-lx'], asl.df['grnd-ly'])
#
#features_delta = ['delta-rx', 'delta-ry', 'delta-lx', 'delta-ly']
#
#diffs = asl.df.groupby(asl.df.index.get_level_values(0)).diff().fillna(0)
#for feature_delta in features_delta:
#    side_axis = feature_delta.split('-')[-1]
#    asl.df["delta-{}".format(side_axis)] = diffs["grnd-{}".format(side_axis)]
#
#features_custom = ['norm-rr', 'norm-rtheta', 'norm-lr', 'norm-ltheta']
#
#for feature_custom in features_custom:
#    side_axis = feature_custom.split('-')[-1]
#    asl.df[feature_custom] = std("polar-{}".format(side_axis), asl.df, df_means, df_std)
#    
#features_custom2 = ['custom-rx', 'custom-ry', 'custom-lx', 'custom-ly']
#for feature_custom in features_custom2:
#    side_axis = feature_custom.split('-')[-1]
#    asl.df[feature_custom] = std("grnd-{}".format(side_axis), asl.df, df_means, df_std)


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
