from basic_pytorch.gpu_utils import FloatTensor, to_gpu, LongTensor
import torch
from torch.autograd import Variable
import numpy as np

class GenericCodec:
    def encode(self, smiles):
        one_hot = self.string_to_one_hot(smiles)
        z_mean = self.vae.encoder.encode(one_hot)
        if type(z_mean) == tuple:
            z_mean = z_mean[0]
        return z_mean

    def decode(self, z):
        '''
        Converts a batch of latent vectors into a batch of action ints
        :param z: batch x z_size
        :return: smiles: list(str) of len batch, actions: LongTensor batch_size x max_seq_len
        '''
        actions, logits = self.decoder(to_gpu(z))
        smiles = self.decode_from_actions(actions)
        return smiles, actions

    def decode_with_validation(self, z, max_attempts = 10):
        import rdkit
        if type(z) == np.ndarray:
            numpy_output = True
            z = FloatTensor(z)
        else:
            numpy_output=False

        out = []
        actions = []
        for this_z in z:
            for _ in range(max_attempts):
                smiles, action = self.decode(torch.unsqueeze(this_z,0))
                result = rdkit.Chem.MolFromSmiles(smiles[0])
                if result is not None:
                    break
            out.append(smiles[0])
            actions.append(action)
        actions = torch.cat(actions, axis=0)
        if numpy_output:
            actions = actions.cpu().numpy()
        return out, actions

    def action_seq_length(self,these_actions):
        if 'numpy' in str(type(these_actions)):
            # TODO: put a numpy-specific version here, not needing pytorch
            these_actions = LongTensor(these_actions)
        out = LongTensor((len(these_actions)))
        for i in range(len(these_actions)):
            out[i] = torch.nonzero(these_actions[i] == (self._n_chars -1))[0]
        return out

def to_one_hot(y, n_dims=None, out = None):
    """
    Take integer y (tensor or variable) with n dims and convert it to 1-hot representation with n+1 dims.
    The one-hot dimension is added at the end
    Taken from https://discuss.pytorch.org/t/convert-int-into-one-hot-format/507/24?u=egor_kraev
    """
    y_tensor = y.data if isinstance(y, Variable) else y
    y_tensor = y_tensor.type(LongTensor).view(-1, 1)
    n_dims = n_dims if n_dims is not None else int(torch.max(y_tensor)) + 1
    y_one_hot = torch.zeros(y_tensor.size()[0], n_dims, out=out).scatter_(1, y_tensor, 1)
    y_one_hot = y_one_hot.view(*y.shape, -1)
    return Variable(y_one_hot) if isinstance(y, Variable) else y_one_hot