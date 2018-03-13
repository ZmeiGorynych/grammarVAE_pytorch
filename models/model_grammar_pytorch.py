from collections import OrderedDict
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from basic_pytorch.gpu_utils import FloatTensor, IntTensor, to_gpu
from basic_pytorch.models.rnn_models import SimpleRNNDecoder,SimpleRNNAttentionEncoder


class Encoder(nn.Module):
    def __init__(self,
                 max_seq_length=None,
                 encoder_kernel_sizes=(2,3,4),
                 z_size=200,
                 feature_len=None,
                 drop_rate = 0.0):
        super(Encoder, self).__init__()
        self.k = encoder_kernel_sizes
        # NOTE: GVAE implementation does not use max-pooling. Original DCNN implementation uses max-k pooling.
        conv_args = {'in_channels':feature_len, 'out_channels':feature_len,  'groups':feature_len}
        self.dropout1 = nn.Dropout(drop_rate)
        self.conv_1 = nn.Conv1d(kernel_size=self.k[0],**conv_args)
        self.bn_1 = nn.BatchNorm1d(feature_len)
        self.dropout2 = nn.Dropout(drop_rate)

        self.conv_2 = nn.Conv1d(kernel_size=self.k[1],**conv_args)
        self.bn_2 = nn.BatchNorm1d(feature_len)
        self.dropout3 = nn.Dropout(drop_rate)
        self.conv_3 = nn.Conv1d(kernel_size=self.k[2],**conv_args)
        self.bn_3 = nn.BatchNorm1d(feature_len)
        self.dropout4 = nn.Dropout(drop_rate)

        self.fc_0 = nn.Linear(feature_len * (max_seq_length + len(self.k) - sum(self.k)), z_size)
        self.dropout5 = nn.Dropout(drop_rate)
        self.fc_mu = nn.Linear(z_size, z_size)
        self.fc_var = nn.Linear(z_size, z_size)

    def forward(self, x):
        batch_size = x.size()[0]
        # Conv1D expects dimension batch x channels x feature
        # we treat the one-hot encoding as channels, but only convolve one channel at a time?
        # why not just view() the array into the right shape?
        x = x.transpose(1, 2)#.contiguous()
        x = self.dropout1(x)
        x = F.relu(self.bn_1(self.conv_1(x)))
        x = self.dropout2(x)
        x = F.relu(self.bn_2(self.conv_2(x)))
        x = self.dropout3(x)
        x = F.relu(self.bn_3(self.conv_3(x)))
        x = self.dropout4(x)
        x_ = x.view(batch_size, -1)
        h = self.fc_0(x_)
        h = self.dropout5(h)
        return self.fc_mu(h), self.fc_var(h)

    def encode(self,x):
        '''

        :param x: a numpy array batch x seq x feature
        :return:
        '''
        mu_, var_ = self.forward(Variable(FloatTensor(x)))
        return mu_.data.cpu().numpy(), var_.data.cpu().numpy()

#from visdom_helper.visdom_helper import Dashboard


class VAELoss(nn.Module):
    # matches the impelentation in model_eq.py
    def __init__(self, grammar=None, sample_z=False):
        '''
        :param masks: array of allowed transition rules from a given symbol
        '''
        super(VAELoss, self).__init__()
        self.sample_z = sample_z
        self.bce_loss = nn.BCELoss(size_average = False)
        self.masks = FloatTensor(grammar.masks)
        self.ind_to_lhs_ind = IntTensor(grammar.ind_to_lhs_ind)

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
        # added normalization by seq_len as KL term too weak otherwise
        # if torch.max(model_out_x) > 1.0 or torch.min(model_out_x) <0.0:
        #     pass
        model_out_x = F.softmax(model_out_x, dim=2)
        #BCE = -(nn.LogSoftmax(dim=2)(model_out_x)*target_x).sum()/(seq_len * batch_size)
        BCE = self.bce_loss(model_out_x, target_x) / (seq_len * batch_size)
        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        avg_mu = torch.sum(mu, dim=0) / batch_size
        var = torch.mm(mu.t(), mu) / batch_size
        var_err = var - Variable(to_gpu(torch.eye(z_size)))
        var_err = F.tanh(var_err)*var_err # so it's ~ x^2 asymptotically, not x^4
        mom_err = (avg_mu * avg_mu).sum() / z_size + var_err.sum() / (z_size * z_size)
        if self.sample_z:
            KLD_element = mu.pow(2).add_(log_var.exp()).mul_(-1).add_(1).add_(log_var)
            KLD = torch.sum(KLD_element).mul_(-0.5)/batch_size
            KLD_ = KLD.data[0]
            my_loss = BCE + KLD
        else:
            my_loss = BCE + mom_err
            KLD_ = 0
        if not self.training:
            # ignore regularizers when computing validation loss
            my_loss = BCE

        self.metrics =OrderedDict([('BCE', BCE.data[0]),
                                   ('KLD', KLD_),
                                   ('ME', mom_err.data[0])])
        #print(self.metrics)
        return my_loss

def apply_masks(x_true, x_pred, masks, ind_to_lhs_ind):
    '''
    Apply grammar transition rules to a softmax matrix
    :param x_true: Variable of actual transitions, one-hot encoded, batch x sequence x element
    :param x_pred: Variable of logits, same shape as x_true
    :return: x_pred zeroed out and rescaled
    '''

    x_size = x_true.size()
    mask = to_gpu(torch.ones(*x_size))
    # adding this to an element will move it to at least min - 100
    shift_to_tiny = -100 + (x_pred.min() - x_pred.max())
    for i in range(0,x_size[0]):
        for j in range(0, x_size[1]):
            # argmax
            true_rule_ind = torch.max(x_true.data[i,j,:],0)[1][0]
            # look up lhs from true one-hot, mask must be for that lhs
            mask[i,j,:] = masks[ind_to_lhs_ind[true_rule_ind]]

    # nuke the transitions prohibited if we follow x_true
    x_resc = x_pred + ( 1 - Variable(mask))*shift_to_tiny
    # and rescale the softmax to sum=1 again
    #scaler = torch.sum(x_resc, dim=2, keepdim=True)
    #scaler2 = torch.cat([scaler]*x_size[2], dim=2)
    out = x_resc #/(scaler2 + 1e-6)
    return out


class GrammarVariationalAutoEncoder(nn.Module):
    def __init__(self, z_size=200,
                 hidden_n=200,
                 feature_len=12,
                 max_seq_length=15,
                 encoder_kernel_sizes=(2,3,4),
                 drop_rate = 0.0,
                 sample_z = True,
                 rnn_encoder = False):
        super(GrammarVariationalAutoEncoder, self).__init__()
        if rnn_encoder:
            sample_z = False
        self.sample_z = sample_z
        if rnn_encoder:
            self.encoder = to_gpu(SimpleRNNAttentionEncoder(max_seq_length=max_seq_length,
                                                     z_size=z_size,
                                                     hidden_n=hidden_n,
                                                     feature_len=feature_len,
                                                     drop_rate = drop_rate))
        else:
            self.encoder = to_gpu(Encoder(encoder_kernel_sizes=encoder_kernel_sizes,
                                          max_seq_length=max_seq_length,
                                          z_size=z_size,
                                          feature_len=feature_len,
                                          drop_rate=drop_rate))

        self.decoder = to_gpu(SimpleRNNDecoder(z_size=z_size,
                                               hidden_n=hidden_n,
                                               feature_len=feature_len,
                                               max_seq_length=max_seq_length,
                                               drop_rate=drop_rate))

    def forward(self, x):
        batch_size = x.size()[0]
        mu, log_var = self.encoder(x)
        # only sample when training, I regard sampling as a regularization technique
        if self.sample_z and self.training:
            z = self.sample(mu, log_var)
        else:
            z = mu
        h1 = self.decoder.init_hidden(batch_size)
        output, h1 = self.decoder(z, h1)
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
