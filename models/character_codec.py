import numpy as np
from grammarVAE_pytorch.models.reinforcement.reinforcement import SimpleDiscreteDecoder, OneStepDecoderContinuous
from grammarVAE_pytorch.models.policy import SoftmaxRandomSamplePolicy
from grammarVAE_pytorch.models.codec import GenericCodec
import torch

class CharacterModel(GenericCodec):
    def __init__(self,
                 model = None,
                 max_len = None,
                 charlist = None
                 ):
        self.charlist = charlist
        # below is the shared code
        self.MAX_LEN = max_len
        self._char_index = {}
        for ix, char in enumerate(self.charlist):
            self._char_index[char] = ix
        #self.mask_gen = None
        if model is not None:
            self.vae = model
            self.vae.eval()
            self.decoder = self.vae.decoder
        # policy = SoftmaxRandomSamplePolicy()
        # stepper = OneStepDecoderContinuous(self.vae.decoder)
        # self.decoder = SimpleDiscreteDecoder(stepper, policy, self.mask_gen)

    def string_to_one_hot(self, smiles):
        """ Encode a list of smiles strings into the latent space """
        indices = [np.array([self._char_index[c] for c in entry], dtype=int) for entry in smiles]
        one_hot = np.zeros((len(indices), self.MAX_LEN, len(self.charlist)), dtype=np.float32)
        for i in range(len(indices)):
            num_productions = len(indices[i])
            one_hot[i][np.arange(num_productions),indices[i]] = 1.
            one_hot[i][np.arange(num_productions, self.MAX_LEN),-1] = 1.
        return one_hot

    # # TODO: move to superclass?
    # def encode(self, smiles):
    #     one_hot = self.string_to_one_hot(smiles)
    #     z_mean = self.vae.encoder.encode(one_hot)
    #     if type(z_mean) == tuple:
    #         z_mean = z_mean[0]
    #     return z_mean
    #
    # def latent_to_actions(self, z):
    #     """ Sample from the grammar decoder """
    #     assert z.ndim == 2
    #
    #     if type(z) == np.ndarray:
    #         numpy_output = True
    #         z = FloatTensor(z)
    #     else:
    #         numpy_output=False
    #
    #     actions, logits = self.decoder(z)
    #     if numpy_output:
    #         actions = actions.cpu().numpy()
    #         logits = logits.detach().cpu().numpy()
    #     return actions, logits

    def decode_from_actions(self, actions):
        char_matrix = np.array(self.charlist)[np.array(actions, dtype=int)]
        return [''.join(ch).strip() for ch in char_matrix]

    # TODO: move to superclass
    # def decode(self, z):
    #     actions, logits = self.latent_to_actions(z)
    #     smiles = self.decode_from_actions(actions)
    #     return smiles, actions
    #
    # # TODO: move to superclass; works only for molecules
    # def decode_with_validation(self, z, max_attempts = 10):
    #     import rdkit
    #     if type(z) == np.ndarray:
    #         numpy_output = True
    #         z = FloatTensor(z)
    #     else:
    #         numpy_output=False
    #
    #     out = []
    #     actions = []
    #     for this_z in z:
    #         for _ in range(max_attempts):
    #             smiles, action = self.decode(torch.unsqueeze(this_z,0))
    #             result = rdkit.Chem.MolFromSmiles(smiles[0])
    #             if result is not None:
    #                 break
    #         out.append(smiles[0])
    #         actions.append(action)
    #     actions = torch.cat(actions, axis=0)
    #     if numpy_output:
    #         actions = actions.cpu().numpy()
    #     return out, actions


