import networkx as nx
from rdkit.Chem import Descriptors, rdmolops
from rdkit.Chem.rdmolfiles import MolFromSmiles

from bayesian_opt import sascorer


def get_score_components(smiles):
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