import numpy as np
from model_utils import many_one_hot
import h5py

L = []
chars = ['C', '(', ')', 'c', '1', '2', 'o', '=', 'O', 'N', '3', 'F', '[', '@', 'H', ']', 'n', '-', '#', 'S', 'l', '+', 's', 'B', 'r', '/', '4', '\\', '5', '6', '7', 'I', 'P', '8', ' ']
DIM = len(chars)

with open('data/250k_rndm_zinc_drugs_clean.smi','r') as f:
    for line in f:
        line = line.strip()
        L.append(line)

count = 0
MAX_LEN = 120
OH = np.zeros((len(L),MAX_LEN,DIM))
for chem in L:
    indices = []
    for c in chem:
        indices.append(chars.index(c))
    if len(indices) < MAX_LEN:
        indices.extend((MAX_LEN-len(indices))*[DIM-1])
    OH[count,:,:] = many_one_hot(np.array(indices), DIM)
    count = count + 1

with h5py.File('data/zinc_str_dataset.h5','w') as h5f:
    h5f.create_dataset('data', data=OH, compression="gzip", compression_opts=5)
    h5f.create_dataset('chr',  data=np.array(chars, dtype = 'S'))

