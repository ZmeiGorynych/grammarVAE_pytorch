import os, inspect
import numpy as np
import h5py
try:
    import grammarVAE_pytorch
except:
    import os, inspect, sys
    my_location = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    sys.path.append('../..')

from grammarVAE_pytorch.models.model_settings import get_settings, get_model
from basic_pytorch.data_utils.data_sources import IncrementingHDF5Dataset
# change this to true to produce the equation dataset
molecules = True
# change this to True to get string-based encodings instead of grammar-based
grammar = True

# can't define model class inside settings as it itself uses settings a lot
_, my_model = get_model(molecules,grammar)
settings = get_settings(molecules,grammar)
MAX_LEN = settings['max_seq_length']
feature_len = settings['feature_len']
dest_file = settings['data_path']
source_file = settings['source_data']

# Read in the strings
f = open(source_file,'r')
L = []
for line in f:
    line = line.strip()
    L.append(line)
f.close()

# convert to one-hot and save, in small increments to save RAM
dest_file = dest_file.replace('.h5','_new.h5')
ds = IncrementingHDF5Dataset(dest_file)


# TODO: for molecules, also append the score
# TODO: make sure this also works for char-based models
step = 100
for i in range(0, len(L), step):#for i in range(0, 1000, 2000):
    print('Processing: i=[' + str(i) + ':' + str(i + step) + ']')
    these_smiles = L[i:min(i + step,len(L))]
    these_actions = my_model.string_to_actions(these_smiles)
    action_seq_length = my_model.action_seq_length(these_actions)
    onehot = my_model.actions_to_one_hot(these_actions)
    append_data = {'smiles': np.array(these_smiles, dtype='S'),
                   'actions': these_actions,
                   'valid': np.ones((len(these_smiles))),
                   'seq_len': action_seq_length,
                   'data': onehot}
    ds.append(append_data)

print('success!')

