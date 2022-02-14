import uproot as up
import numpy as np
import h5py
import pandas as pd
from variable_mapping import mapping
from keras.utils import np_utils
import plot_library

import argparse

parser = argparse.ArgumentParser(description='options for converting files')
parser.add_argument('-i', '--inputfile', type=str,
		default = '/afs/cern.ch/work/c/cainswor/public/DL/tt_jets_NN_input.root')


parser.add_argument('-o', '--outputfile', type=str, default = 'output_convert/with_weights.h5')
args = parser.parse_args()

Signal = up.open(args.inputfile)['NN_signal']
Background = up.open(args.inputfile)['NN_background']

var_list = list(mapping.keys())

weight_list_S = Signal.pandas.df("weights")
weight_list_B = Background.pandas.df("weights")

df_S = Signal.pandas.df(var_list)
df_B = Background.pandas.df(var_list)

## plotting input variables distributions 
plot_library.variable_plotting(df_S, df_B)

## mixing signal and background events
X_train = np.concatenate((pd.DataFrame(df_B), pd.DataFrame(df_S)))
labels = np.concatenate((np.zeros(len(df_B), dtype=int), np.ones(len(df_S), dtype=int)))
weights = np.concatenate((pd.DataFrame(weight_list_B), pd.DataFrame(weight_list_S)))
weights = weights.flatten()
Y_train = np_utils.to_categorical(labels, 2, dtype=int)
print(X_train.shape)
print(weights)
print(labels)
rng_state = np.random.get_state()
np.random.shuffle(X_train)
np.random.set_state(rng_state)
np.random.shuffle(Y_train)
np.random.set_state(rng_state)
np.random.shuffle(labels)
np.random.set_state(rng_state)
np.random.shuffle(weights)
assert X_train.shape[1] == len(var_list)

outputfile = h5py.File(args.outputfile,'w')
outputfile.create_dataset('X_train', data=X_train, compression='gzip')
outputfile.create_dataset('Y_train', data=Y_train, compression='gzip')
outputfile.create_dataset('labels', data=labels, compression='gzip')
outputfile.create_dataset('weights', data=weights, compression='gzip')
outputfile.close()
