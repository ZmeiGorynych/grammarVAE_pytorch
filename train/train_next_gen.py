#!/home/carnd/anaconda3/envs/torch/bin/python

# One upside for calling this as shell script rather than as 'python x.py' is that
# you can see the script name in top/ps - useful when you have a bunch of python processes

try:
    import grammarVAE_pytorch
except:
    import sys, os, inspect
    my_location = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    sys.path.append('../..')

from basic_pytorch.visdom_helper.visdom_helper import Dashboard
from grammarVAE_pytorch.train.main_train_vae import train_vae
from grammarVAE_pytorch.models.model_settings import settings_zinc as settings
from grammarVAE_pytorch.models.grammar_ed_models import ZincGrammarModel, fraction_valid
from basic_pytorch.models.simple_models import DenseHead
import numpy as np
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from basic_pytorch.gpu_utils import FloatTensor, to_gpu

dash_name = 'test'
visdom = Dashboard(dash_name)
model, fitter = train_vae(molecules=True,
                          BATCH_SIZE=50,
                          drop_rate=0.5,
                          sample_z=False,
                          save_file='dropout_no_sampling_rnn_encoder.h5',
                          rnn_encoder=True,
                          lr=1e-4,
                          plot_prefix='RNN enc lr 1e-4',
                          dashboard=dash_name,
                          preload_weights=False)
# this is a wrapper for encoding/decodng
grammar_model = ZincGrammarModel(model=model)
validity_model = to_gpu(DenseHead(model.encoder, body_out_dim=settings['z_size']))
count = 1
sm_metrics = [0,0,0]
f_valid = open("valid_1.smi", "a+")
f_invalid = open("invalid_1.smi",'a+')
# h5invalid = HDF5Dataset('invalid_smiles.h5', max_len = ...)
while True:
    # this does one train step
    count +=1
    next(fitter)
    if True: #count % 10 == 0:
        mock_latent_points = np.random.normal(size=(100,settings['z_size']))
        mock_smiles = grammar_model.decode(mock_latent_points, validate=True, max_attempts=3)
        mock_smiles = [s for s in mock_smiles if s != '']
        metrics, (valid,invalid) = fraction_valid(mock_smiles) # frac_valid, avg_len, max_len
        sm_metrics = [0.9*sm + 0.1*m for sm,m in zip(sm_metrics,metrics)]
        visdom.append('molecule validity',
                   'line',
                   X=np.array([count]),
                   Y=np.array([sm_metrics]),
                   opts={'legend': ['num_valid','avg_len','max_len']})
        # one_hot = grammar_model.string_to_one_hot(mock_smiles)
        # one_hot = Variable(FloatTensor(one_hot))
        # x = validity_model(one_hot)
        #invalid_one_hot = grammar_model.string_to_one_hot(invalid)
        #h5invalid.circular_overwrite('one_hot',invalid_one_hot, max_len = ..)
        #h5invalid.circular_overwrite('smiles',invalid, max_len = ..)

        # for s in valid:
        #     f_valid.write(s+"\n")
        #
        # for s in invalid:
        #     f_invalid.write(s+"\n")

