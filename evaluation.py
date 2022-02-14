import numpy as np
import h5py
import tensorflow as tf
from plot_lib import plotOutputScore
import argparse

parser = argparse.ArgumentParser(
    description = 'options for running evaluation'
)
parser.add_argument('-i', '--input_file',type=str, default = "SamplePrep/output_convert/with_weights.h5",
    help='file used to do evaluation'
)
parser.add_argument('-m', '--model_name',type=str, default = "output_train/training_b512_e150_bestacc.h5",
    help='trained model'
)
parser.add_argument('-o', '--output',type=str,default = "output/",
    help='output file'
)

args = parser.parse_args()

h5f_test = h5py.File(args.input_file, 'r')

X_test = h5f_test['X_train'][:]
labels = h5f_test['labels'][:]
weights = h5f_test['weights'][:]
h5f_test.close()

test_model = tf.keras.models.load_model(args.model_name)
output = test_model.predict(X_test, verbose=2)

plotOutputScore(output[:,0], labels, weights, args.output)

h5f = h5py.File('{}/evaluaiton.h5'.format(args.output), 'w')
h5f.create_dataset('outputScore', data=output[:,0], compression='gzip')
h5f.create_dataset('labels', data=labels, compression='gzip')
h5f.create_dataset('weights', data=weights, compression='gzip')
h5f.close()
