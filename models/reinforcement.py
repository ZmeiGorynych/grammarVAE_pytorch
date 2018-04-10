import torch
from torch.autograd import Variable
import torch.nn as nn
from torch.nn import functional as F
from basic_pytorch.gpu_utils import FloatTensor, to_gpu
from grammarVAE_pytorch.models.decoders import OneStepDecoderContinuous, SimpleDiscreteDecoder
from grammarVAE_pytorch.models.policy import SoftmaxRandomSamplePolicy, PolicyFromTarget


# TODO: model already outputs values!
class ReinforcementModel(nn.Module):
    '''
    Creates targets from a one-step iteration of the Bellman equation
    '''
    def __init__(self, discrete_decoder: SimpleDiscreteDecoder, disc_factor=0.99):
        super().__init__()
        self.decoder = discrete_decoder
        self.disc_factor = disc_factor

    def forward(self,inputs):
        '''
        Calculates targets and values for simple deepQ-learning
        :param z: latent variable input for decoder, batch_size x z_size
        :param actions: History of actions, batch_size x max_len of ints
        :param sample_ind: Index we want to sample each sequence at, batch_size
        :param values: Value of objective function where sequence comleted, None otherwise : batch_size
        :return: values: value of action at sample_ind predicted by decoder value; targets: one-step Bellman equation target
        '''
        actions, seq_len, score, sample_ind = inputs
        batch_size = len(actions)
        orig_policy = self.decoder.policy
        self.decoder.policy = PolicyFromTarget(actions)
        _, logits = self.decoder.forward(to_gpu(torch.zeros(batch_size,self.decoder.z_size)))
        # todo: a better head, now values and policy too entangled
        value_est = F.tanh(logits)
        targets = Variable(FloatTensor(batch_size))
        # values currently estimated by the network
        est_values = Variable(FloatTensor(batch_size))
        for n in range(batch_size):
            # the value of the action chosen at the sampled step for that sequence
            est_values[n] = value_est[n,
                                      sample_ind[n],
                                      actions[n,sample_ind[n]]]
            if sample_ind[n]+1 < seq_len[n] is None: # so the sequence is not complete at this point
                # one step
                targets[n] = self.disc_factor*torch.max(logits[n, sample_ind[n]+1, :])
            else:
                targets[n] = score[n]
        targets.detach()
        self.decoder.policy = orig_policy
        return est_values, targets

class ReinforcementLoss(nn.Module):
    def forward(self, outputs, targets):
        '''
        A reinforcement learning loss. As both predicted Q and target Q estimated as max from next step
        come from the model itself, the 'targets' are ignored
        :param outputs:
        :param targets:
        :return:
        '''
        predicted_Q, target_Q = outputs
        diff = predicted_Q - target_Q
        return torch.mean(diff * diff)


