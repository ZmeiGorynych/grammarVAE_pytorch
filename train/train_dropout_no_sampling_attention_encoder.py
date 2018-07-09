#!/home/carnd/anaconda3/envs/torch/bin/python

# One upside for calling this as shell script rather than as 'python x.py' is that
# you can see the script name in top/ps - useful when you have a bunch of python processes

try:
    import grammarVAE_pytorch
except:
    import sys, os, inspect
    my_location = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    sys.path.append('../..')
    sys.path.append('../../DeepRL')
    sys.path.append('../../generative_playground')
    sys.path.append('../../transformer_pytorch')

from grammarVAE_pytorch.train.main_train_vae import train_vae

from generative_playground.models.model_settings import get_settings

molecules = True
grammar = False
settings = get_settings(molecules,grammar)

save_file =settings['filename_stub'] + 'dr0.2_attention.h5'

model, fitter, train_dataset = train_vae(molecules=molecules,
                                         grammar=grammar,
                          BATCH_SIZE=30,
                          drop_rate=0.2,
                          save_file=save_file,
                          sample_z=False,
                          rnn_encoder='attention', # cnn, rnn, attention
                          decoder_type='attention',
                          lr=1e-4,
                          plot_prefix='attn do=0.2 no_sam 1e-4')

while True:
    next(fitter)

