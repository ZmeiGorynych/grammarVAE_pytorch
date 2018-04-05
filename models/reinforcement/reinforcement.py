import torch
from torch.autograd import Variable
import torch.nn as nn
from basic_pytorch.gpu_utils import FloatTensor, to_gpu
from grammarVAE_pytorch.models.codec import to_one_hot
from grammarVAE_pytorch.models.policy import SimplePolicy, SoftmaxRandomSamplePolicy, PolicyFromTarget


class OneStepDecoder(nn.Module):
    '''
    One step of a decoder into a discrete space, suitable for use with autoencoders
    (so the encoded state is a vector not a sequence)

    '''
    def __init__(self, model, max_len = None):
        '''
        Base class for doing the differentiable part of one decoding step
        :param model: a differentiable model used in the steps
        '''
        super().__init__()
        self.n = 0
        self.model = to_gpu(model)
        self.model.eval()
        self.max_len = max_len

    def init_latent(self, z):
        '''
        Start decoding a new batch
        :param z: batch_size x num actions or batch_size x max_input_len x num_actions encoded state
        :return: None
        '''
        self.z = z
        self.n = 0
        try:
            self.model.reset_state()
        except:
            pass

    def forward(self, action):
        '''
        # the differentiable part of one decoding step
        :param action: LongTensor((batch_size)), last discrete action chosen by the policy,
        None for the very first action choice
        :return: FloatTensor((batch_size x num_actions)), an unmasked vector of logits over next actions
        '''
        if self.n < self.max_len:
            out = self.model(self.z)
            out = torch.squeeze(out,1)
            self.n += 1
            return out
        else:
            raise StopIteration()

class OneStepDecoderUsingAction(OneStepDecoder):
    def __init__(self, model, max_len=None, num_actions = None):
        '''
        Base class for doing the differentiable part of one decoding step
        :param model: a differentiable model used in the steps
        '''
        super().__init__(model, max_len)
        self.num_actions = num_actions

    def init_latent(self, z):
        super().init_latent(z)
        self.one_hot_action = to_gpu(torch.zeros(len(self.z), self.num_actions))

    def forward(self, action):
        '''
        # the differentiable part of one decoding step
        :param action: LongTensor((batch_size)), last discrete action chosen by the policy,
        None for the very first action choice
        :return: FloatTensor((batch_size x num_actions)), an unmasked vector of logits over next actions
        '''
        if self.n < self.max_len:
            if True:#type(action)== list or len(action.shape) == 1: # if the actions came encoded as ints
                if action is not None and action[0] is not None: # if not first call
                    self.one_hot_action = to_one_hot(action,
                                                n_dims=self.num_actions,
                                                out = self.one_hot_action)
                model_input = torch.cat([self.z,self.one_hot_action], 1)
            # else: # if the actions already came encoded as one-hot
            #     model_input = torch.cat([self.z, action], 1)
            out = self.model(model_input)
            out = torch.squeeze(out, 1)
            self.n += 1
            return out
        else:
            raise StopIteration()


class OneStepDecoderContinuous(OneStepDecoder):
    def __init__(self,model):
        '''
        Implementation for a continuous decoder that doesn't look at last action chosen, eg simple RNN
        :param model:
        '''
        super().__init__(model)

    def init_latent(self, z):
        super().init_latent(z)
        self.logits = self.model.forward(z)

    def forward(self, action=None):
        '''

        :param action: ignored
        :return: a vector of logits over next actions
        '''
        if self.n < self.logits.shape[1]:
            out = torch.squeeze(self.logits[:, self.n, :],1)
            self.n +=1
            return out
        else:
            raise StopIteration()


class DummyMaskGenerator(nn.Module):
    def __init__(self, num_actions):
        super().__init__()
        self.num_actions = num_actions

    def forward(self, last_action):
        '''
        Consumes one action at a time, responds with the mask for next action
        : param last_action: ints of shape (batch_size) previous action ; should be [None]*batch_size for the very first step
        '''
        return to_gpu(torch.ones(len(last_action),self.num_actions))

    def reset(self):
        '''
        Reset any internal state, in order to start on a new sequence
        :return:
        '''
        pass



class SimpleDiscreteDecoder(nn.Module):
    def __init__(self, stepper:OneStepDecoder, policy: SimplePolicy, mask_gen = None, bypass_actions=False):
        '''
        A simple discrete decoder, alternating getting logits from model and actions from policy
        :param stepper:
        :param policy: choose an action from the logits, can be max, or random sample,
        or choose from pre-determined target sequence. Only depends on current logits + history,
        can't handle multi-step strategies like beam search
        :param mask_fun: takes in one-hot encoding of previous action (for now that's all we care about)
        '''
        super().__init__()
        self.stepper = to_gpu(stepper)
        self.policy = policy
        self.mask_gen = mask_gen
        self.bypass_actions = bypass_actions

    def forward(self, z):
        # initialize the decoding model
        self.stepper.init_latent(z)
        if self.bypass_actions:
            return None, self.stepper.logits
        out_logits = []
        out_actions = []
        last_action = [None]*len(z)
        step = 0
        # as it's PyTorch, can determine max_len dynamically, by when the stepper raises StopIteration
        while True:
            try:
  #          if True:
                # dimension batch x num_actions
                next_logits = self.stepper(last_action)
                if self.mask_gen is not None:
                    # mask_gen might return a numpy mask
                    mask = FloatTensor(self.mask_gen(last_action))
                    masked_logits = next_logits - 1e4*(1-mask)
                else:
                    masked_logits = next_logits

                next_action = self.policy(masked_logits)
                out_logits.append(torch.unsqueeze(masked_logits,1))
                out_actions.append(torch.unsqueeze(next_action,1))
                last_action = next_action
            except StopIteration as e:
                #print(e)
                break
        if self.mask_gen is not None:
            self.mask_gen.reset()
        out_actions_all = torch.cat(out_actions, 1)
        out_logits_all = torch.cat(out_logits, 1)
        return out_actions_all, out_logits_all


# TODO: model already outputs values!
class ReinforcementModel(nn.Module):
    '''
    Creates targets from a one-step iteration of the Bellman equation
    '''
    def __init__(self, discrete_decoder: SimpleDiscreteDecoder):
        super().__init__()
        self.decoder = discrete_decoder

    def forward(self, z, actions, sample_ind, values):
        '''
        Calculates targets and values for simple deepQ-learning
        :param z: latent variable input for decoder, batch_size x z_size
        :param actions: History of past decoding actions, batch_size x max_len x num_actions
        :param sample_ind: Index we want to sample each sequence at, batch_size
        :param values: Value of objective function where sequence comleted, None otherwise : batch_size
        :return: values: value of action at sample_ind predicted by decoder value; targets: one-step Bellman equation target
        '''
        batch_size = len(z)
        self.decoder.policy = PolicyFromTarget(actions)
        _, logits = self.decoder.forward(z)
        targets = Variable(FloatTensor(batch_size))
        values = Variable(FloatTensor(batch_size))
        for n in range(batch_size):
            # the value of the action chosen at the sampled step for that sequence
            values[n] = (logits[n,sample_ind[n]]*actions[n,sample_ind[n],:]).sum()
            if values[n] is None:
                # one step
                targets[n] = torch.max(logits[n, sample_ind[n]+1, :])
            else:
                targets[n] = values[n]


        return values, targets

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
        target_Q.detach()
        diff = predicted_Q - target_Q
        return torch.sum(diff * diff)

class DummyNNModel(nn.Module):
    def __init__(self, max_len, num_actions):
        super().__init__()
        self.max_len = max_len
        self.num_actions = num_actions

    def forward(self, z):
        return to_gpu(torch.randn(len(z), self.max_len, self.num_actions))

if __name__ == '__main__':
    batch_size = 25
    max_len = 10
    num_actions = 15
    latent_size = 20

    policy = SoftmaxRandomSamplePolicy()
    mask_gen = DummyMaskGenerator(num_actions)
    stepper = OneStepDecoderContinuous(DummyNNModel(max_len, num_actions))
    decoder = SimpleDiscreteDecoder(stepper,policy,mask_gen)
    z = torch.randn(batch_size, latent_size)
    out_actions, out_logits = decoder(z)
    print('success!')