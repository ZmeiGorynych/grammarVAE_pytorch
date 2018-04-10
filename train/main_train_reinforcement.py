import os, inspect
import torch.optim as optim
from torch.optim import lr_scheduler

from basic_pytorch.fit import fit
from basic_pytorch.data_utils.data_sources import train_valid_loaders, SamplingWrapper
from basic_pytorch.gpu_utils import use_gpu
from grammarVAE_pytorch.models.model_settings import get_settings
from grammarVAE_pytorch.data_utils.mixed_loader import MixedLoader, CombinedLoader
from grammarVAE_pytorch.models.reinforcement import ReinforcementLoss

def train_reinforcement(grammar = True,
              model = None,
              EPOCHS = None,
              BATCH_SIZE = None,
              lr = 2e-4,
              main_dataset = None,
              new_datasets = None,
              plot_ignore_initial = 0,
              save_file = None,
              plot_prefix = '',
              dashboard = 'main',
              preload_weights=False):

    root_location = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    root_location = root_location + '/../'
    if save_file is not None:
        save_path = root_location + 'pretrained/' + save_file
    else:
        save_path = None
    molecules = True # checking for validity only makes sense for molecules
    settings = get_settings(molecules=molecules,grammar=grammar)

    # TODO: separate settings for this?
    if EPOCHS is not None:
        settings['EPOCHS'] = EPOCHS
    if BATCH_SIZE is not None:
        settings['BATCH_SIZE'] = BATCH_SIZE

    if preload_weights:
        try:
            model.load(save_path)
        except:
            pass

    optimizer = optim.Adam(model.parameters(), lr=lr)

    # # create the composite loaders
    # train_loader, valid_loader = train_valid_loaders(main_dataset,
    #                                                  valid_fraction=0.1,
    #                                                  batch_size=BATCH_SIZE,
    #                                                  pin_memory=use_gpu)
    train_l=[]
    valid_l=[]
    for ds in new_datasets:
        train_loader, valid_loader = SamplingWrapper(ds)\
                        .get_train_valid_loaders(BATCH_SIZE,
                            dataset_name=['actions','seq_len','valid','sample_seq_ind'])
        train_l.append(train_loader)
        valid_l.append(valid_loader)
    train_gen = CombinedLoader(train_l, num_batches=100)
    valid_gen = CombinedLoader(valid_l, num_batches=10)

    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer,
                                               factor=0.2,
                                               patience=3,
                                               min_lr=0.0001,
                                               eps=1e-08)
    #scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)
    loss_obj = ReinforcementLoss()

    fitter = fit(train_gen=train_gen,
        valid_gen=valid_gen,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        epochs=settings['EPOCHS'],
        loss_fn=loss_obj,
        save_path=save_path,
        dashboard=dashboard,
        plot_ignore_initial=plot_ignore_initial,
                 plot_prefix=plot_prefix)

    return fitter


