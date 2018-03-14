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

model, fitter = train_vae(molecules=True,
          BATCH_SIZE=50,
          drop_rate=0.5,
          sample_z=False,
          save_file='dummy.h5',#''dropout_no_sampling_rnn_encoder_.h5',
          rnn_encoder=True,
          lr=2e-3,
          plot_prefix='RNN enc high LR')

while True:
    next(fitter)
