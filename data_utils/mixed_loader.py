import torch
from basic_pytorch.gpu_utils import FloatTensor, to_gpu
class MixedLoader:
    def __init__(self, main_loader, valid_ds, invalid_ds):
        self.main_loader = main_loader
        self.valid_ds = valid_ds
        self.invalid_ds = invalid_ds
        self.batch_size = main_loader.batch_size

    def __len__(self):
        return len(self.main_loader)

    def __iter__(self):
        def gen():
            iter1 = iter(self.main_loader)
            iter2 = iter(self.valid_ds)
            iter3 = iter(self.invalid_ds)
            while True:
                # make sure we iterate fully over the first dataset, others will likely be shorter
                x1 = next(iter1).float()
                try:
                    x2 = next(iter2).float()
                except StopIteration:
                    iter2 = iter(self.valid_ds)
                    x2 = next(iter2).float()
                try:
                    x3 = next(iter3).float()
                except StopIteration:
                    iter3 = iter(self.valid_ds)
                    x3 = next(iter3).float()

                x = to_gpu(torch.cat([x1,x2,x3], dim=0))
                y = to_gpu(torch.zeros([len(x),1]))
                y[:(len(x1)+len(x2))]=1
                yield x,y

        return gen()