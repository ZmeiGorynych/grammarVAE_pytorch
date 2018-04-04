# import sys
# sys.path.append('..')
import os, inspect
import numpy as np
from grammarVAE_pytorch.models.model_settings import get_settings, get_model
from basic_pytorch.gpu_utils import to_gpu
import torch

# We load the auto-encoder
my_location = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
for molecules in [True, False]:
    for grammar in [True, False]:
        model, wrapper_model = get_model(molecules=molecules, grammar=grammar)
        settings = get_settings(molecules=molecules, grammar=grammar)
        for _ in range(10):
            z = to_gpu(torch.randn(1,settings['z_size']))
            new_smile, _ = wrapper_model.decode(z)
            print(new_smile)