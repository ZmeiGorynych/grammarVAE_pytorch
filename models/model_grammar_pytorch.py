import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from basic_pytorch.gpu_utils import FloatTensor, to_gpu
from basic_pytorch.models.rnn_models import SimpleRNNDecoder,SimpleRNNAttentionEncoder


class Encoder(nn.Module):
    def __init__(self,
                 max_seq_length=None,
                 params = {},
                 z_size=200,
                 feature_len=None,
                 drop_rate = 0.0):
        super(Encoder, self).__init__()
        self.k = params['kernel_sizes']
        self.ch = params['filters']
        self.dense_size = params['dense_size']
        # NOTE: GVAE implementation does not use max-pooling. Original DCNN implementation uses max-k pooling.
        #conv_args = {'in_channels':feature_len, 'out_channels':feature_len,  'groups':feature_len}
        self.dropout1 = nn.Dropout(drop_rate)
        self.conv_1 = nn.Conv1d(kernel_size=self.k[0],
                                in_channels=feature_len,
                                out_channels=self.ch[0])
        self.bn_1 = nn.BatchNorm1d(self.ch[0])
        self.dropout2 = nn.Dropout(drop_rate)

        self.conv_2 = nn.Conv1d(kernel_size=self.k[1],
                                in_channels=self.ch[0],
                                out_channels=self.ch[1])
        self.bn_2 = nn.BatchNorm1d(self.ch[1])
        self.dropout3 = nn.Dropout(drop_rate)
        self.conv_3 = nn.Conv1d(kernel_size=self.k[2],
                                in_channels=self.ch[1],
                                out_channels=self.ch[2])
        self.bn_3 = nn.BatchNorm1d(self.ch[2])
        self.dropout4 = nn.Dropout(drop_rate)

        self.fc_0 = nn.Linear(self.ch[2] * (max_seq_length + len(self.k) - sum(self.k)), self.dense_size)
        self.dropout5 = nn.Dropout(drop_rate)
        self.fc_mu = nn.Linear(self.dense_size, z_size)
        self.fc_var = nn.Linear(self.dense_size, z_size)

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
        h = F.relu(self.fc_0(x_))
        h = self.dropout5(h)
        return self.fc_mu(h), self.fc_var(h)

    def encode(self,x):
        '''

        :param x: a numpy array batch x seq x feature
        :return:
        '''
        mu_, var_ = self.forward(Variable(FloatTensor(x)))
        return mu_.data.cpu().numpy()


class GrammarVariationalAutoEncoder(nn.Module):
    def __init__(self, z_size=200,
                 decoder_hidden_n=200,
                 feature_len=12,
                 max_seq_length=15,
                 cnn_encoder_params={'kernel_sizes': (2, 3, 4),
                                               'filters': (2, 3, 4),
                                               'dense_size': 100},
                 drop_rate = 0.0,
                 sample_z = True,
                 rnn_encoder = False,
                 rnn_encoder_hidden_n = 200):
        super(GrammarVariationalAutoEncoder, self).__init__()
        if rnn_encoder:
            sample_z = False
        self.sample_z = sample_z
        if rnn_encoder:
            self.encoder = to_gpu(SimpleRNNAttentionEncoder(max_seq_length=max_seq_length,
                                                     z_size=z_size,
                                                     hidden_n=rnn_encoder_hidden_n,
                                                     feature_len=feature_len,
                                                     drop_rate = drop_rate))
        else:
            self.encoder = to_gpu(Encoder(params=cnn_encoder_params,
                                          max_seq_length=max_seq_length,
                                          z_size=z_size,
                                          feature_len=feature_len,
                                          drop_rate=drop_rate))

        self.decoder = to_gpu(SimpleRNNDecoder(z_size=z_size,
                                               hidden_n=decoder_hidden_n,
                                               feature_len=feature_len,
                                               max_seq_length=max_seq_length,
                                               drop_rate=drop_rate))

    def forward(self, x):
        batch_size = x.size()[0]
        mu, log_var = self.encoder(x)
        # only sample when training, I regard sampling as a regularization technique so unneeded during validation
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
