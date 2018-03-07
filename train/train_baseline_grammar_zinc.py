#!/home/carnd/anaconda3/bin/python

try:
    import grammarVAE_pytorch
except:
    import sys, os, inspect
    my_location = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    sys.path.append('../..')

from train.main_train_vae import train_vae
train_vae(molecules=True)
