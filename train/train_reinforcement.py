#!/home/carnd/anaconda3/envs/torch/bin/python

# One upside for calling this as shell script rather than as 'python x.py' is that
# you can see the script name in top/ps - useful when you have a bunch of python processes
import sys
try:
    import grammarVAE_pytorch
except:
    import os, inspect
    my_location = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    sys.path.append('../..')
import os
# # fix for a PyCharm bug for DLL import
# os.environ['CONDA_PREFIX'] = 'C:/Users/egork/Anaconda3/envs/torch/'
# from subprocess import call
# call(["C:/Users/egork/Anaconda3/envs/torch/Scripts/activate.bat", 'torch'])
# sys.path.append('C:/Users/egork/Anaconda3/envs/torch/DLLs')
# sys.path.append('C:/Users/egork/Anaconda3/envs/torch/Scripts')
# sys.path.append('C:\\Users\\egork\\Anaconda3\\envs\\torch\\Lib\\site-packages\\rdkit')
import torch
from basic_pytorch.data_utils.data_sources import IncrementingHDF5Dataset
from basic_pytorch.visdom_helper.visdom_helper import Dashboard
from grammarVAE_pytorch.train.main_train_vae import train_vae
from grammarVAE_pytorch.train.main_train_reinforcement import train_reinforcement
from grammarVAE_pytorch.models.model_settings import get_settings, get_model
from grammarVAE_pytorch.models.grammar_codec import GrammarModel, eq_tokenizer, zinc_tokenizer
from grammarVAE_pytorch.models.reinforcement import ReinforcementModel
from grammarVAE_pytorch.models.rdkit_utils import fraction_valid
import numpy as np

molecules = True
grammar = True
settings = get_settings(molecules, grammar)

dash_name = 'test'
visdom = Dashboard(dash_name)
model, grammar_model = get_model(molecules, grammar, drop_rate=0.3)
reinforcement_model = ReinforcementModel(model.decoder)
valid_smile_ds = IncrementingHDF5Dataset('valid_smiles.h5')
invalid_smile_ds = IncrementingHDF5Dataset('invalid_smiles.h5')
original_ds = IncrementingHDF5Dataset('../data/zinc_grammar_dataset.h5')

RL_fitter = train_reinforcement(grammar = grammar,
              model = reinforcement_model,
              EPOCHS = 100,
              BATCH_SIZE = 40,
              lr = 1e-4,
              new_datasets = (valid_smile_ds, invalid_smile_ds, original_ds),
              save_file = None,
              plot_prefix = 'valid_model',
              dashboard = dash_name,
              preload_weights=False)

count = 0
sm_metrics = None
have_visdom = True
while True:
    # this does one train step
    #next(fitter)
    mock_latent_points = torch.zeros(size=(100,settings['z_size']))#np.random.normal(size=(100,settings['z_size']))
    mock_smiles, mock_actions = grammar_model.decode(mock_latent_points)
    action_seq_length = grammar_model.action_seq_length(mock_actions)
    mock_actions = mock_actions.cpu().numpy()
    action_seq_length = action_seq_length.cpu().numpy()

    if len([s for s in mock_smiles if s == '']):
        raise ValueError("With the new masking, sequences should always complete!")

    metrics, mols = fraction_valid(mock_smiles) # frac_valid, avg_len, max_len
    is_valid = [mol is not None for mol in mols]
    if sm_metrics is None:
        sm_metrics = metrics
    else:
        sm_metrics = [0.9*sm + 0.1*m for sm,m in zip(sm_metrics,metrics)]

    if have_visdom:
        try:
            visdom.append('molecule validity',
                       'line',
                       X=np.array([count]),
                       Y=np.array([sm_metrics]),
                       opts={'legend': ['num_valid','avg_len','max_len']})
        except:
            have_visdom = False

    for condition, ds in (True, valid_smile_ds), (False, invalid_smile_ds):
        these_smiles = [smile for smile, valid in zip(mock_smiles,is_valid) if valid==condition]
        these_actions = mock_actions[np.array(is_valid) == condition]
        this_len = action_seq_length[np.array(is_valid) == condition]
        append_data ={'smiles': np.array(these_smiles, dtype = 'S'),
                      'actions': these_actions,
                      'valid': np.ones((len(these_smiles)))*(1 if condition else 0),
                      'seq_len': this_len}
        ds.append(append_data)

    next(RL_fitter)
    count +=1


        # for s in valid:
        #     f_valid.write(s+"\n")
        #
        # for s in invalid:
        #     f_invalid.write(s+"\n")

