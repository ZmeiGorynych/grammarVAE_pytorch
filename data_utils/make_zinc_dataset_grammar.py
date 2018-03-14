from __future__ import print_function

from models.model_settings import settings_zinc as settings
from grammarVAE_pytorch.models.grammar_ed_models import ZincGrammarModel as ThisGrammarModel
import numpy as np
import h5py

MAX_LEN=settings['max_seq_length']
NCHARS = settings['feature_len']
grammar_model = ThisGrammarModel()
data_dir = '../data/'
# Read in the SMILES strings
f = open(data_dir + '250k_rndm_zinc_drugs_clean.smi','r')
L = []
#count = -1
for line in f:
    line = line.strip()
    L.append(line)
f.close()

# convert to one-hot and save
dataset_created = False
with h5py.File(data_dir + 'zinc_grammar_dataset.h5', 'w') as h5f:
    for i in range(0, len(L), 1000):
        print('Processing: i=[' + str(i) + ':' + str(i+1000) + ']')
        onehot = grammar_model.smiles_to_one_hot(L[i:i+1000])
        if not dataset_created:
            h5f.create_dataset('data', data=onehot,
                               compression="gzip",
                               compression_opts=9,
                               maxshape = (len(L),MAX_LEN,NCHARS))
            dataset_created = True
        else:
            h5f["data"].resize(h5f["data"].shape[0] + onehot.shape[0], axis=0)
            h5f["data"][-onehot.shape[0]:] = onehot

