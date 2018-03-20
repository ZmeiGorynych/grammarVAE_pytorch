import os, inspect
import numpy as np
import h5py
from models.model_settings import get_settings

# change this to true to produce the equation dataset
molecules = True
# change this to True to get string-based encodings instead of grammar-based
grammar = True

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
dataset_created = False
with h5py.File(dest_file, 'w') as h5f:
    for i in range(0, len(L), 1000):
        print('Processing: i=[' + str(i) + ':' + str(i+1000) + ']')
        onehot = my_model.string_to_one_hot(L[i:i + 1000])
        if not dataset_created:
            h5f.create_dataset('data', data=onehot,
                               compression="gzip",
                               compression_opts=9,
                               maxshape = (len(L), MAX_LEN, feature_len))
            dataset_created = True
        else:
            h5f["data"].resize(h5f["data"].shape[0] + onehot.shape[0], axis=0)
            h5f["data"][-onehot.shape[0]:] = onehot

print('success!')
