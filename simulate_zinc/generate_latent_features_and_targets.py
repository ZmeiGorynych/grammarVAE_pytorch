from rdkit.Chem import Descriptors
from rdkit.Chem import rdmolops
from rdkit.Chem import MolFromSmiles, MolToSmiles
import sascorer
import numpy as np  
import os, inspect
import networkx as nx

from grammarVAE_pytorch.models import grammar_ed_models as grammar_model

# We load the auto-encoder
# my_location = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
# grammar_weights = my_location + '/pretrained/my_molecules.mdl'
# grammar_model = grammar_model.ZincGrammarModel(grammar_weights)
# z = grammar_model.encode(['c1nccc2n1ccc2'])
# print(type(z))
# new_smile = grammar_model.decode(z)
# print(new_smile)
# We load the smiles data
from models import grammar_ed_models as grammar_model
my_location = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
fname = my_location + '/../data/250k_rndm_zinc_drugs_clean.smi'
# mock_latent_points = np.random.normal(size=latent_points.shape)
# mock_smiles = grammar_model.decode(mock_latent_points)
# mock_mols = []

# for m in mock_smiles:
#     mock_mols.append(MolFromSmiles(m))
smiles_rdkit = []
with open(fname) as f:
    smiles = f.readlines()

smiles = [s.strip() for s in smiles]
#    smiles[i] = smiles[i].strip()

# We load the auto-encoder
max_len = len(smiles) #10

import sys
sys.path.insert(0, my_location + '/../')

grammar_weights = my_location + '/../pretrained/dropout_no_sampling_rnn_encoder.h5'
grammar_model = grammar_model.ZincGrammarModel(grammar_weights, rnn_encoder=True)

for i in range(max_len):#:
    smiles_rdkit.append(MolToSmiles(MolFromSmiles(smiles[i])))
    if i%1000==0 and i>0:
        print(i)
print('Encoding the molecules...')
latent_points = grammar_model.encode(smiles_rdkit)
print('Calculating the scores...')
logP_values = []
SA_scores = []
cycle_scores = []
for i in range(max_len):
    logP_values.append(Descriptors.MolLogP(MolFromSmiles(smiles_rdkit[i])))
    SA_scores.append(-sascorer.calculateScore(MolFromSmiles(smiles_rdkit[i])))
    cycle_list = nx.cycle_basis(nx.Graph(rdmolops.GetAdjacencyMatrix(MolFromSmiles(smiles_rdkit[ i ]))))
    if len(cycle_list) == 0:
        cycle_length = 0
    else:
        cycle_length = max([len(j) for j in cycle_list])
    if cycle_length <= 6:
        cycle_length = 0
    else:
        cycle_length = cycle_length - 6
    cycle_scores.append(-cycle_length)
    if i%1000==0 and i>0:
        print(i)

SA_scores_normalized = (np.array(SA_scores) - np.mean(SA_scores)) / np.std(SA_scores)
logP_values_normalized = (np.array(logP_values) - np.mean(logP_values)) / np.std(logP_values)
cycle_scores_normalized = (np.array(cycle_scores) - np.mean(cycle_scores)) / np.std(cycle_scores)

# We store the results
latent_points = np.array(latent_points)
np.savetxt('latent_features.txt', latent_points)

targets = SA_scores_normalized + logP_values_normalized + cycle_scores_normalized
np.savetxt('targets.txt', targets)
np.savetxt('logP_values.txt', np.array(logP_values))
np.savetxt('SA_scores.txt', np.array(SA_scores))
np.savetxt('cycle_scores.txt', np.array(cycle_scores))
