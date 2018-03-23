from rdkit.Chem.rdmolfiles import MolFromSmiles

def fraction_valid(smiles):
    mols = mol_from_smiles(smiles)
    valid_lens = [len(m.GetAtoms()) for m in mols if m is not None]
    num_valid = len(valid_lens)
    avg_len = sum(valid_lens) / (len(valid_lens) + 1e-6)
    max_len = 0 if not len(valid_lens) else max(valid_lens)
    return (num_valid, avg_len, max_len), mols

def mol_from_smiles(smiles):
    if type(smiles)=='str':
        return MolFromSmiles(smiles)
    else: # assume we have a list-like
        return [MolFromSmiles(s) for s in smiles]

