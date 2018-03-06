import h5py
import numpy as np
import os, inspect
import torch.utils.data
import torch.optim as optim
from torch.autograd import Variable
from torch.optim import lr_scheduler
try:
    import grammarVAE_pytorch
except:
    import sys
    sys.path.append('..')

from grammarVAE_pytorch.models.grammar_helper import grammar_eq, grammar_zinc
from grammarVAE_pytorch.models.model_grammar_pytorch import GrammarVariationalAutoEncoder, VAELoss
from basic_pytorch.fit import fit
from basic_pytorch.data_utils.data_sources import DatasetFromHDF5, train_valid_loaders
from basic_pytorch.gpu_utils import to_gpu, use_gpu

my_location = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
molecules = True
if molecules:
    grammar = grammar_zinc
    data_path = 'data/zinc_grammar_dataset.h5'
    save_path = my_location + '/pretrained/my_molecules.mdl'
    max_seq_length = 277
    LATENT_SIZE = 56
    EPOCHS = 100
    BATCH_SIZE = 850  # the most that the 12GB GPU on p2.xlarge will take
else:
    grammar = grammar_eq
    data_path = 'data/eq2_grammar_dataset.h5'
    save_path = my_location + '/pretrained/my_equations.mdl'
    max_seq_length = 15
    LATENT_SIZE = 25
    EPOCHS = 50 # from mkusner
    BATCH_SIZE = 600 # from mkusner
    # TODO: what's the learning rate they use?

model_args = {'z_size': LATENT_SIZE,
              'hidden_n': 200,
              'feature_len': len(grammar.GCFG.productions()),
              'max_seq_length': max_seq_length,
              'encoder_kernel_sizes': (2, 3, 4)}

model = GrammarVariationalAutoEncoder(**model_args)
optimizer = optim.Adam(model.parameters(), lr=2e-3)

class DuplicateIter:
    def __init__(self, iterable):
        self.iterable = iterable

    def __iter__(self):
        def gen():
            iter = self.iterable.__iter__()
            while True:
                # TODO: cast to float earlier?
                x = Variable(to_gpu(next(iter).float()))
                yield (x,x)
        return gen()


train_loader, valid_loader = train_valid_loaders(DatasetFromHDF5(data_path,'data'),
                                                 valid_fraction=0.5,
                                                 batch_size=BATCH_SIZE,
                                                 pin_memory=use_gpu)

train_gen = DuplicateIter(train_loader)
valid_gen = DuplicateIter(valid_loader)

scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)

loss_obj = VAELoss(grammar)
def loss_fn(model_out, data):
    output, mu, log_var = model_out
    return loss_obj(data, mu, log_var, output)



fit(train_gen=train_gen,
    valid_gen=valid_gen,
    model=model,
    optimizer=optimizer,
    scheduler=scheduler,
    epochs=EPOCHS,
    loss_fn=loss_fn,
    save_path=save_path,
    dashboard= "My dashboard",
    ignore_initial=10,
    save_every=100)

# test the Load method
model2 = GrammarVariationalAutoEncoder(**model_args)
model2.load(save_path)
# TODO: use

