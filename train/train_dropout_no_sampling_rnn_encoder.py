#!/home/carnd/anaconda3/envs/torch/bin/python

# One upside for calling this as shell script rather than as 'python x.py' is that
# you can see the script name in top/ps - useful when you have a bunch of python processes

try:
    import grammarVAE_pytorch
except:
    import sys, os, inspect
    my_location = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    sys.path.append('../..')
from grammarVAE_pytorch.train.main_train_vae import train_vae

from grammarVAE_pytorch.models.model_settings import get_settings

molecules = True
grammar = True
settings = get_settings(molecules,grammar)

save_file =settings['filename_stub'] + 'dr0.5_rnnenc_no_sampl.h5'

model, fitter = train_vae(molecules=molecules,
                          BATCH_SIZE=150,
                          drop_rate=0.4,
                          save_file=save_file,
                          sample_z=False,
                          rnn_encoder=True,
                          lr=5e-4,
                          plot_prefix='rnn do=0.3 no_sam 5e-4')

while True:
    next(fitter)

