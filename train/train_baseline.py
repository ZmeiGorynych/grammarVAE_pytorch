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
from generative_playground.models.model_settings import get_settings

molecules = True
grammar = True
settings = get_settings(molecules,grammar)

save_file =settings['filename_stub'] + 'baseline.h5'
model, fitter, _ = train_vae(molecules=molecules,
                          grammar=grammar,
                          BATCH_SIZE=50, # a p2.xlarge won't bear any bigger batches
                          save_file=save_file,
                          sample_z=True,
                          rnn_encoder=False,
                          lr=5e-4,
                          plot_prefix='baseline lr 5e-4 KLW 0.01',
                          KL_weight = 1,
                             epsilon_std = 0.01,
                             dashboard='main')

while True:
    next(fitter)

