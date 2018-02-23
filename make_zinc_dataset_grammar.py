from __future__ import print_function
import nltk
#import pdb
import zinc_grammar
import numpy as np
import h5py
import zinc_tokenizer



MAX_LEN=277
NCHARS = len(zinc_grammar.GCFG.productions())

def to_one_hot(smiles):
    """ Encode a list of smiles strings to one-hot vectors """
    assert type(smiles) == list
    prod_map = {}
    for ix, prod in enumerate(zinc_grammar.GCFG.productions()):
        prod_map[prod] = ix
    tokenize = zinc_tokenizer.get_zinc_tokenizer(zinc_grammar.GCFG)
    tokens = map(tokenize, smiles)
    parser = nltk.ChartParser(zinc_grammar.GCFG)
    parse_trees = [next(parser.parse(t)) for t in tokens]
    productions_seq = [tree.productions() for tree in parse_trees]
    indices = [np.array([prod_map[prod] for prod in entry], dtype=int) for entry in productions_seq]
    one_hot = np.zeros((len(indices), MAX_LEN, NCHARS), dtype=np.float32)
    for i in range(len(indices)):
        num_productions = len(indices[i])
        one_hot[i][np.arange(num_productions),indices[i]] = 1.
        one_hot[i][np.arange(num_productions, MAX_LEN),-1] = 1.
    return one_hot

f = open('data/250k_rndm_zinc_drugs_clean.smi','r')
L = []

count = -1
for line in f:
    line = line.strip()
    L.append(line)
f.close()

OH = []
dataset_created = False
with h5py.File('data/zinc_grammar_dataset.h5', 'w') as h5f:
    for i in range(0, len(L), 100):
        print('Processing: i=[' + str(i) + ':' + str(i+100) + ']')
        onehot = to_one_hot(L[i:i+100])
        OH.append(onehot)
        if i%1000 == 0:
            print('Saving.... ', i )
            OHc = np.concatenate(OH, axis=0)
            OH = []
            if not dataset_created:
                h5f.create_dataset('data', data=OHc,
                                   compression="gzip",
                                   compression_opts=9,
                                   maxshape = (len(L),MAX_LEN,NCHARS))
                dataset_created = True
            else:
                h5f["data"].resize(h5f["data"].shape[0] + OHc.shape[0], axis=0)
                h5f["data"][-OHc.shape[0]:] = OHc

# OH = np.concatenate(OH,axis=0)
#
# h5f = h5py.File('zinc_grammar_dataset.h5','w')
# h5f.create_dataset('data', data=OH)
# h5f.close()
