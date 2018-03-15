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

molecules = False

if molecules:
    save_file = 'grammar_zinc_baseline.h5'
else:
    save_file = 'grammar_eq_baseline.h5'


model, fitter = train_vae(molecules=molecules,
                          BATCH_SIZE=50,
                          save_file=save_file,
                          sample_z=True,
                          rnn_encoder=False,
                          lr=1e-3)

while True:
    next(fitter)

