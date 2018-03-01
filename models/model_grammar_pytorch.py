import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from gpu_utils import FloatTensor, IntTensor, to_gpu

class Decoder(nn.Module):
    # implementation matches model_eq.py _buildDecoder, at least in intent
    def __init__(self, input_size=200, hidden_n=200, output_feature_size=12, max_seq_length=15):
        super(Decoder, self).__init__()
        self.max_seq_length = max_seq_length
        self.hidden_n = hidden_n
        self.output_feature_size = output_feature_size
        self.batch_norm = nn.BatchNorm1d(input_size)
        self.fc_input = nn.Linear(input_size, hidden_n)
        # we specify each layer manually, so that we can do teacher forcing on the last layer.
        # we also use no drop-out in this version.
        self.gru_1 = nn.GRU(input_size=input_size, hidden_size=hidden_n, batch_first=True)
        self.gru_2 = nn.GRU(input_size=input_size, hidden_size=hidden_n, batch_first=True)
        self.gru_3 = nn.GRU(input_size=input_size, hidden_size=hidden_n, batch_first=True)
        self.fc_out = nn.Linear(hidden_n, output_feature_size)

    def forward(self, encoded, hidden_1, hidden_2, hidden_3, beta=0.3, target_seq=None):
        _batch_size = encoded.size()[0]

        embedded = F.relu(self.fc_input(self.batch_norm(encoded))) \
            .view(_batch_size, 1, -1) \
            .repeat(1, self.max_seq_length, 1)
        # batch_size, seq_length, hidden_size; batch_size, hidden_size
        out_1, hidden_1 = self.gru_1(embedded, hidden_1)
        out_2, hidden_2 = self.gru_2(out_1, hidden_2)
        # NOTE: need to combine the input from previous layer with the expected output during training.
        # NOTE: this bit is not in the Keras GrammarVAE code
        if self.training and target_seq:
            out_2 = out_2 * (1 - beta) + target_seq * beta
        out_3, hidden_3 = self.gru_3(out_2, hidden_3)
        out = self.fc_out(out_3.contiguous().view(-1, self.hidden_n)).view(_batch_size, self.max_seq_length,
                                                                           self.output_feature_size)
        # WTF RELU(sigmoid)?
        #return F.relu(F.sigmoid(out)), hidden_1, hidden_2, hidden_3
        return F.softmax(out,2), hidden_1, hidden_2, hidden_3

    def decode(self, z):
        # TODO: should actually be handling single vectors, not batches, dependent on dim?
        batch_size = z.size()[0]
        h1, h2, h3 = self.decoder.init_hidden(batch_size)
        output, h1, h2, h3 = self.decoder(z, h1, h2, h3)
        return output.data.numpy()

    def init_hidden(self, batch_size):
        # NOTE: assume only 1 layer no bi-direction
        h1 = Variable(torch.zeros(1, batch_size, self.hidden_n), requires_grad=False)
        h2 = Variable(torch.zeros(1, batch_size, self.hidden_n), requires_grad=False)
        h3 = Variable(torch.zeros(1, batch_size, self.hidden_n), requires_grad=False)
        return h1, h2, h3


class Encoder(nn.Module):
    def __init__(self, L, k1=2, k2=3, k3=4, hidden_n=200):
        super(Encoder, self).__init__()
        # NOTE: GVAE implementation does not use max-pooling. Original DCNN implementation uses max-k pooling.
        self.conv_1 = nn.Conv1d(in_channels=12, out_channels=12, kernel_size=k1, groups=12)
        self.bn_1 = nn.BatchNorm1d(12)
        self.conv_2 = nn.Conv1d(in_channels=12, out_channels=12, kernel_size=k2, groups=12)
        self.bn_2 = nn.BatchNorm1d(12)
        self.conv_3 = nn.Conv1d(in_channels=12, out_channels=12, kernel_size=k3, groups=12)
        self.bn_3 = nn.BatchNorm1d(12)

        # todo: harded coded because I can LOL
        self.fc_0 = nn.Linear(12 * 9, hidden_n)
        self.fc_mu = nn.Linear(hidden_n, hidden_n)
        self.fc_var = nn.Linear(hidden_n, hidden_n)

    def forward(self, x):
        batch_size = x.size()[0]
        x = x.transpose(1, 2).contiguous()
        x = F.relu(self.bn_1(self.conv_1(x)))
        x = F.relu(self.bn_2(self.conv_2(x)))
        x = F.relu(self.bn_3(self.conv_3(x)))
        x_ = x.view(batch_size, -1)
        h = self.fc_0(x_)
        return self.fc_mu(h), self.fc_var(h)

    def encode(self,x):
        mu_, var_ = self.forward(x)
        return mu_.data.numpy(), var_.data.numpy()

#from visdom_helper.visdom_helper import Dashboard


class VAELoss(nn.Module):
    # matches the impelentation in model_eq.py
    def __init__(self, grammar  = None):
        '''
        :param masks: array of allowed transition rules from a given symbol
        '''
        super(VAELoss, self).__init__()
        self.bce_loss = nn.BCELoss()
        self.bce_loss.size_average = False
        self.masks = FloatTensor(grammar.masks)
        self.ind_to_lhs_ind = IntTensor(grammar.ind_to_lhs_ind)
        #self.dashboard = Dashboard('Variational-Autoencoder-experiment')

    # question: how is the loss function using the mu and variance?
    def forward(self, x, mu, log_var, recon_x):
        """gives the batch normalized Variational Error."""

        batch_size = x.size()[0]
        if self.masks is not None:
            recon_x = apply_masks(x,
                                  recon_x,
                                  self.masks,
                                  self.ind_to_lhs_ind
                                  )

        BCE = self.bce_loss(recon_x, x)

        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD_element = mu.pow(2).add_(log_var.exp()).mul_(-1).add_(1).add_(log_var)
        KLD = torch.sum(KLD_element).mul_(-0.5)

        return (BCE + KLD) / batch_size

def apply_masks(x_true, x_pred, masks, ind_to_lhs_ind):
    '''
    Apply grammar transition rules to a softmax matrix
    :param x_true: Variable of actual transitions, one-hot encoded, batch x sequence x element
    :param x_pred: Variable of probabilities, past softmax, same shape as x_true
    :return: x_pred zeroed out and rescaled
    '''

    x_size = x_true.size()
    mask = to_gpu(torch.ones(*x_size))
    for i in range(0,x_size[0]):
        for j in range(0, x_size[1]):
            # argmax
            true_rule_ind = torch.max(x_true.data[i,j,:],0)[1][0]
            # look up lhs from true one-hot, mask must be for that lhs
            mask[i,j,:] = masks[ind_to_lhs_ind[true_rule_ind]]

    # nuke the transitions prohibited if we follow x_true
    x_resc = x_pred * Variable(mask)
    # and rescale the softmax to sum=1 again
    scaler = torch.sum(x_resc, dim=2, keepdim=True)
    scaler2 = torch.cat([scaler]*x_size[2], dim=2)
    out = x_resc /scaler2
    return out

class GrammarVariationalAutoEncoder(nn.Module):
    def __init__(self):
        super(GrammarVariationalAutoEncoder, self).__init__()
        self.encoder = Encoder(15)
        self.decoder = Decoder()

    def forward(self, x):
        batch_size = x.size()[0]
        mu, log_var = self.encoder(x)
        z = self.sample(mu, log_var)
        h1, h2, h3 = self.decoder.init_hidden(batch_size)
        output, h1, h2, h3 = self.decoder(z, h1, h2, h3)
        return output, mu, log_var

    def sample(self, mu, log_var):
        """you generate a random distribution w.r.t. the mu and log_var from the embedding space."""
        vector_size = log_var.size()
        eps = Variable(torch.FloatTensor(vector_size).normal_())
        std = log_var.mul(0.5).exp_()
        return eps.mul(std).add_(mu)

    def load(self,
             productions = None,
             weights_file = None,
             max_length=None,
             latent_rep_size=None):
        raise NotImplementedError()
