from collections import OrderedDict

import torch
from torch import nn as nn
from torch.autograd import Variable
import torch.nn as nn
from torch.nn import functional as F

from basic_pytorch.gpu_utils import FloatTensor, LongTensor, to_gpu


class GrammarVariationalAutoEncoder(nn.Module):
    def __init__(self, encoder = None,
                 decoder = None,
                 sample_z = True):
        super(GrammarVariationalAutoEncoder, self).__init__()
        self.sample_z = sample_z
        self.encoder = to_gpu(encoder)
        self.decoder = to_gpu(decoder)

    def forward(self, x):
        #batch_size = x.size()[0]
        mu, log_var = self.encoder(x)
        # only sample when training, I regard sampling as a regularization technique so unneeded during validation
        if self.sample_z and self.training:
            z = self.sample(mu, log_var)
        else:
            z = mu
        #h1 = self.decoder.init_hidden(batch_size)
        actions, output = self.decoder(z)
        return output, mu, log_var

    def sample(self, mu, log_var):
        """you generate a random distribution w.r.t. the mu and log_var from the embedding space."""
        vector_size = log_var.size()
        eps = Variable(FloatTensor(vector_size).normal_())
        std = log_var.mul(0.5).exp_()
        return eps.mul(std).add_(mu)

    def load(self, weights_file):
        print('Trying to load model parameters from ', weights_file)
        self.load_state_dict(torch.load(weights_file))
        self.eval()
        print('Success!')


class VAELoss(nn.Module):
    # matches the impelentation in model_eq.py
    def __init__(self, grammar=None, sample_z=False, KL_weight = 0.01):
        '''
        :param masks: array of allowed transition rules from a given symbol
        '''
        super(VAELoss, self).__init__()
        self.sample_z = sample_z
        self.bce_loss = nn.BCELoss(size_average = True) #following mkusner/grammarVAE, earlier was False)
        self.KL_weight = KL_weight
        if grammar is not None:
            self.masks = FloatTensor(grammar.masks)
            self.ind_to_lhs_ind = LongTensor(grammar.ind_to_lhs_ind)
        else:
            self.masks = None

    def forward(self, model_out, target_x):
        """gives the batch normalized Variational Error."""
        model_out_x, mu, log_var = model_out
        batch_size = target_x.size()[0]
        seq_len = target_x.size()[1]
        z_size = mu.size()[1]
        if self.masks is not None:
            model_out_x = apply_masks(target_x,
                                  model_out_x,
                                  self.masks,
                                  self.ind_to_lhs_ind
                                  )
        model_out_x = F.softmax(model_out_x, dim=2)
        # Was
        #BCE = self.bce_loss(model_out_x, target_x) / (seq_len * batch_size)
        # now like this, following mkusner/grammarVAE
        BCE = seq_len * self.bce_loss(model_out_x, target_x)
        # this normalizer is for when we're not sampling so only have mus, not sigmas
        avg_mu = torch.sum(mu, dim=0) / batch_size
        var = torch.mm(mu.t(), mu) / batch_size
        var_err = var - Variable(to_gpu(torch.eye(z_size)))
        var_err = F.tanh(var_err)*var_err # so it's ~ x^2 asymptotically, not x^4
        mom_err = (avg_mu * avg_mu).sum() / z_size + var_err.sum() / (z_size * z_size)
        if self.sample_z:
            # see Appendix B from VAE paper:
            # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
            # https://arxiv.org/abs/1312.6114
            # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
            KLD_element = (1 + log_var - mu*mu - log_var.exp())
            KLD = -0.5 * torch.mean(KLD_element)
            KLD_ = KLD.data.item()
            my_loss = BCE + self.KL_weight * KLD
        else:
            my_loss = BCE + mom_err
            KLD_ = 0
        if not self.training:
            # ignore regularizers when computing validation loss
            my_loss = BCE

        self.metrics =OrderedDict([('BCE', BCE.data.item()),
                                   ('KLD', KLD_),
                                   ('ME', mom_err.data.item())])
                                   # ('FV', FV)])#,
                                   # ('avg_len', avg_len),
                                   # ('max_len',max_len)])
        #print(self.metrics)
        return my_loss


def apply_masks(x_true, x_pred, masks, ind_to_lhs_ind):
    '''
    Apply grammar transition rules to a softmax matrix, given a one-hot target
    :param x_true: Variable of actual transitions, one-hot encoded, batch x sequence x element
    :param x_pred: Variable of logits, same shape as x_true
    :return: x_pred with masked logits shifted down by at least -100 below original min()
    '''

    x_size = x_true.size()
    mask = to_gpu(torch.ones(*x_size))
    # adding this to an element will move it to at least min - 100
    shift_to_tiny = -100 + (x_pred.min() - x_pred.max())
    for i in range(0,x_size[0]):
        for j in range(0, x_size[1]):
            #print(x_true.data[i,j,:])
            # argmax
            _,true_rule_ind = torch.max(x_true.data[i,j,:],-1)#[1][0]
            # look up lhs from true one-hot, mask must be for that lhs
            mask[i,j,:] = masks[ind_to_lhs_ind[true_rule_ind]]

    # nuke the transitions prohibited if we follow x_true
    x_resc = x_pred + ( 1 - Variable(mask))*shift_to_tiny
    # and rescale the softmax to sum=1 again
    #scaler = torch.sum(x_resc, dim=2, keepdim=True)
    #scaler2 = torch.cat([scaler]*x_size[2], dim=2)
    out = x_resc #/(scaler2 + 1e-6)
    return out