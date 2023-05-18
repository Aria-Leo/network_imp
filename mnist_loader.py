import gzip
import pickle

import numpy as np


def load_data(file_path, mode='rb'):
    f = gzip.open(file_path, mode)
    tr_d, va_d, te_d = pickle.load(f, encoding='bytes')
    f.close()
    training_inputs = tr_d[0]
    training_results = np.eye(10)[tr_d[1]]
    tr_d = (training_inputs, training_results)
    return tr_d, va_d, te_d
