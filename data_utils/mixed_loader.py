import torch
from basic_pytorch.gpu_utils import FloatTensor, to_gpu
class MixedLoader:
    def __init__(self, main_loader, valid_ds, invalid_ds):
        self.main_loader = main_loader
        self.valid_ds = valid_ds
        self.invalid_ds = invalid_ds
        self.batch_size = main_loader.batch_size


    def __len__(self):
        return self.num_batches# Wlen(self.main_loader)

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

class MixedLoader2:
    def __init__(self, ds1, ds2, num_batches=100):
        '''
        Glues together output from 2 loaders: goal is to produce a balanced dataset from the 2 sources
        :param ds1:
        :param ds2:
        '''
        self.ds1 = ds1
        self.ds2 = ds2
        self.batch_size = self.ds1.batch_size
        self.num_batches = num_batches

    def __len__(self):
        return self.num_batches

    def __iter__(self):
        def gen():
            iter1 = iter(self.ds1)
            iter2 = iter(self.ds2)
            for _ in range(self.num_batches):
                try:
                    x1 = next(iter1)
                except StopIteration:
                    iter1 = iter(self.ds1)
                    x1 = next(iter1)
                try:
                    x2 = next(iter2)
                except StopIteration:
                    iter2 = iter(self.ds2)
                    x2 = next(iter2)

                if type(x1) == tuple or type(x1) == list:
                    x = tuple(to_gpu(torch.cat([x1_,x2_], dim=0)) for x1_, x2_ in zip(x1,x2))
                else:
                    x = to_gpu(torch.cat([x1,x2], dim=0))
                yield (x, to_gpu(torch.zeros(1))) # inputs, targets: targets are ignored by the reinf learning algo

        return gen()