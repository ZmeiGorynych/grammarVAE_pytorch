import h5py
import numpy as np
import os, inspect
import torch.utils.data
import torch.optim as optim
from torch.autograd import Variable
from torch.optim import lr_scheduler

from grammarVAE_pytorch.models.grammar_helper import grammar_eq, grammar_zinc
from grammarVAE_pytorch.models.model_grammar_pytorch import GrammarVariationalAutoEncoder, VAELoss
from basic_pytorch.fit import fit
from basic_pytorch.data_utils.data_sources import DatasetFromHDF5, train_valid_loaders
from basic_pytorch.gpu_utils import to_gpu, use_gpu

def train_vae(molecules = True,
              EPOCHS = None,
              BATCH_SIZE = None,
              lr = 2e-3,
              drop_rate = 0.0,
              plot_ignore_initial = 10,
              sample_z = True,
              save_file = None):
    root_location = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    root_location = root_location + '/../'
    if molecules:
        grammar = grammar_zinc
        data_path = root_location + 'data/zinc_grammar_dataset.h5'
        if not save_file:
            save_file = 'my_molecules.mdl'
        save_path = root_location + 'pretrained/' + save_file
        max_seq_length = 277
        LATENT_SIZE = 56
        if not EPOCHS:
            EPOCHS = 100
        if not BATCH_SIZE:
            BATCH_SIZE = 850  # the most that the 12GB GPU on p2.xlarge will take
    else:
        grammar = grammar_eq
        data_path = root_location + 'data/eq2_grammar_dataset.h5'
        if not save_file:
            save_file = 'my_equations.mdl'
        save_path = root_location + 'pretrained/' + save_file
        max_seq_length = 15
        LATENT_SIZE = 25
        if not EPOCHS:
            EPOCHS = 50 # from mkusner/grammarVAE
        if not BATCH_SIZE:
            BATCH_SIZE = 600 # from mkusner/grammarVAE
        # TODO: what's the learning rate they use?

    model_args = {'z_size': LATENT_SIZE,
                  'hidden_n': 200,
                  'feature_len': len(grammar.GCFG.productions()),
                  'max_seq_length': max_seq_length,
                  'encoder_kernel_sizes': (2, 3, 4),
                  'drop_rate': drop_rate,
                  'sample_z': sample_z}

    model = GrammarVariationalAutoEncoder(**model_args)
    optimizer = optim.Adam(model.parameters(), lr=lr)

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

        def __len__(self):
            return len(self.iterable)


    train_loader, valid_loader = train_valid_loaders(DatasetFromHDF5(data_path,'data'),
                                                     valid_fraction=0.1,
                                                     batch_size=BATCH_SIZE,
                                                     pin_memory=use_gpu)

    train_gen = DuplicateIter(train_loader)
    valid_gen = DuplicateIter(valid_loader)

    scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)

    loss_obj = VAELoss(grammar, sample_z=sample_z)
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
        plot_ignore_initial=plot_ignore_initial)

# test the Load method
# model2 = GrammarVariationalAutoEncoder(**model_args)
# model2.load(save_path)
# TODO: use

