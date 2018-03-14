from rdkit.Chem import Descriptors
from rdkit.Chem import rdmolops
from rdkit.Chem import MolFromSmiles, MolToSmiles
from data_utils import sascorer
import numpy as np  
import os, inspect
import networkx as nx
import h5py
import sys
my_location = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
sys.path.insert(0, my_location + '/../')
from grammarVAE_pytorch.models import grammar_ed_models as grammar_model

def get_scores(smiles):
    this_mol = MolFromSmiles(smiles)
    logP = Descriptors.MolLogP(this_mol)
    SA_score = -sascorer.calculateScore(this_mol)
    cycle_list = nx.cycle_basis(nx.Graph(rdmolops.GetAdjacencyMatrix(this_mol)))
    if len(cycle_list) == 0:
        cycle_length = 0
    else:
        cycle_length = max([len(j) for j in cycle_list])
    if cycle_length <= 6:
        cycle_length = 0
    else:
        cycle_length = cycle_length - 6
    cycle_score = -cycle_length
    return logP, SA_score, cycle_score
# We load the auto-encoder
# my_location = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
# grammar_weights = my_location + '/pretrained/my_molecules.mdl'
# grammar_model = grammar_model.ZincGrammarModel(grammar_weights)
# z = grammar_model.encode(['c1nccc2n1ccc2'])
# print(type(z))
# new_smile = grammar_model.decode(z)
# print(new_smile)
# mock_latent_points = np.random.normal(size=latent_points.shape)
# mock_smiles = grammar_model.decode(mock_latent_points)
# mock_mols = []

# for m in mock_smiles:
#     mock_mols.append(MolFromSmiles(m))
# We load the smiles data

smiles_fname = my_location + '/../data/250k_rndm_zinc_drugs_clean.smi'
onehot_fname = my_location + '/../data/zinc_grammar_dataset.h5'
data_dir = '../data/'

# load the pre-trained model
grammar_weights = my_location + '/../pretrained/dropout_no_sampling_rnn_encoder.h5'
grammar_model = grammar_model.ZincGrammarModel(grammar_weights, rnn_encoder=True)

smiles_rdkit = []
with open(smiles_fname) as f:
    smiles = f.readlines()

smiles = [s.strip() for s in smiles]
print('validating smiles strings...')
smiles_rdkit = [MolToSmiles(MolFromSmiles(s)) for s in smiles]  # what's the point of that? validation?
print('done!')

block_size = 1000
# note: you need to run make_zinc_dataset_grammar.py first to generate that hdf5 file from .smi
latent_points = []
raw_scores = []
with h5py.File(onehot_fname, 'r') as h5f:
    for block_start in range(0,len(smiles),block_size):
        block_end = min(block_start+block_size,len(smiles))
        print(block_start, ':', block_end)
        # conversion from SMILES to one hot has already been done prior to calibrating the model
        one_hot = h5f['data'][block_start:block_end]
        latent_points.append(grammar_model.vae.encoder.encode(one_hot))
        smiles_chunk = smiles_rdkit[block_start:block_end]
        raw_scores += [get_scores(s) for s in smiles_chunk]

latent_points = np.concatenate(latent_points)
raw_scores = np.array(raw_scores)

logP_values = raw_scores[:,0]
SA_scores = raw_scores[:,1]
cycle_scores = raw_scores[:,2]

SA_scores_normalized = (np.array(SA_scores) - np.mean(SA_scores)) / np.std(SA_scores)
logP_values_normalized = (np.array(logP_values) - np.mean(logP_values)) / np.std(logP_values)
cycle_scores_normalized = (np.array(cycle_scores) - np.mean(cycle_scores)) / np.std(cycle_scores)

# We store the results
with h5py.File(data_dir + 'zinc_grammar_latent_and_scores.h5', 'w') as h5f:

        h5f.create_dataset('scores', data=raw_scores,
                           compression="gzip",
                           compression_opts=9)
        h5f.create_dataset('latent', data=latent_points,
                           compression="gzip",
                           compression_opts=9)


