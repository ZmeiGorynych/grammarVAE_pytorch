from basic_pytorch.gpu_utils import FloatTensor
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
        actions, logits = self.decoder(z)
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

def to_one_hot(y, n_dims=None):
    """
    Take integer y (tensor or variable) with n dims and convert it to 1-hot representation with n+1 dims.
    The one-hot dimension is added at the end
    Taken from https://discuss.pytorch.org/t/convert-int-into-one-hot-format/507/24?u=egor_kraev
    """
    y_tensor = y.data if isinstance(y, Variable) else y
    y_tensor = y_tensor.type(torch.LongTensor).view(-1, 1)
    n_dims = n_dims if n_dims is not None else int(torch.max(y_tensor)) + 1
    y_one_hot = torch.zeros(y_tensor.size()[0], n_dims).scatter_(1, y_tensor, 1)
    y_one_hot = y_one_hot.view(*y.shape, -1)
    return Variable(y_one_hot) if isinstance(y, Variable) else y_one_hot