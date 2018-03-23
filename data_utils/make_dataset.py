import os, inspect
import numpy as np
import h5py
from models.model_settings import get_settings
from basic_pytorch.data_utils.data_sources import IncrementingHDF5Dataset
# change this to true to produce the equation dataset
molecules = False
# change this to True to get string-based encodings instead of grammar-based
grammar = False

# can't define model class inside settings as it itself uses settings a lot
if molecules:
    if grammar:
        from grammarVAE_pytorch.models.grammar_ed_models import ZincGrammarModel as ThisModel
    else:
        from grammarVAE_pytorch.models.character_ed_models import ZincCharacterModel as ThisModel
else:
    if grammar:
        from grammarVAE_pytorch.models.grammar_ed_models import EquationGrammarModel as ThisModel
    else:
        from grammarVAE_pytorch.models.character_ed_models import EquationCharacterModel as ThisModel

settings = get_settings(molecules,grammar)
MAX_LEN = settings['max_seq_length']
feature_len = settings['feature_len']
dest_file = settings['data_path']
source_file = settings['source_data']

my_model = ThisModel()
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
    print('Processing: i=[' + str(i) + ':' + str(i+1000) + ']')
    onehot = my_model.string_to_one_hot(L[i:i + 1000])
    ds.append(onehot)

print('success!')
train_ds, test_ds = ds.get_train_valid_datasets()
len(train_ds)
train_ds[0]

