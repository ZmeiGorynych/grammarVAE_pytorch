import os, inspect
import torch.optim as optim
from torch.optim import lr_scheduler

from grammarVAE_pytorch.models.grammar_helper import grammar_eq, grammar_zinc
from grammarVAE_pytorch.models.model_grammar_pytorch import GrammarVariationalAutoEncoder, VAELoss
from basic_pytorch.fit import fit
from basic_pytorch.data_utils.data_sources import DatasetFromHDF5, train_valid_loaders, DuplicateIter
from basic_pytorch.gpu_utils import use_gpu
from grammarVAE_pytorch.models.model_settings import settings_eq, settings_zinc, get_model_args

def train_vae(molecules = True,
              EPOCHS = None,
              BATCH_SIZE = None,
              lr = 2e-4,
              drop_rate = 0.0,
              plot_ignore_initial = 0,
              sample_z = True,
              save_file = None,
              rnn_encoder=False):
    root_location = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    root_location = root_location + '/../'
    save_path = root_location + 'pretrained/' + save_file

    if molecules:
        settings = settings_zinc
    else:
        settings = settings_eq

    if EPOCHS is not None:
        settings['EPOCHS'] = EPOCHS
    if BATCH_SIZE is not None:
        settings['BATCH_SIZE'] = BATCH_SIZE

    model_args = get_model_args(molecules, drop_rate, sample_z, rnn_encoder)
    model = GrammarVariationalAutoEncoder(**model_args)

    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_loader, valid_loader = train_valid_loaders(DatasetFromHDF5(settings['data_path'],'data'),
                                                     valid_fraction=0.1,
                                                     batch_size=BATCH_SIZE,
                                                     pin_memory=use_gpu)

    train_gen = DuplicateIter(train_loader)
    valid_gen = DuplicateIter(valid_loader)

    scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)
    loss_obj = VAELoss(settings['grammar'], sample_z)

    fitter = fit(train_gen=train_gen,
        valid_gen=valid_gen,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        epochs=settings['EPOCHS'],
        loss_fn=loss_obj,
        save_path=save_path,
        dashboard= "My dashboard",
        plot_ignore_initial=plot_ignore_initial)

    return model, fitter


