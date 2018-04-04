import os, inspect
import numpy as np
import h5py
from grammarVAE_pytorch.models.model_settings import get_settings, get_model
from basic_pytorch.data_utils.data_sources import IncrementingHDF5Dataset
# change this to true to produce the equation dataset
molecules = False
# change this to True to get string-based encodings instead of grammar-based
grammar = False

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
ds = IncrementingHDF5Dataset(dest_file, valid_frac=0.1)

for i in range(0, len(L), 1000):
#for i in range(0, 10000, 1000):
    print('Processing: i=[' + str(i) + ':' + str(i+1000) + ']')
    onehot = my_model.string_to_one_hot(L[i:i + 1000])
    ds.append(onehot)

print('success!')
train_loader, valid_loader = ds.get_train_valid_loaders(batch_size=100)
len(valid_loader)

print('success!')

