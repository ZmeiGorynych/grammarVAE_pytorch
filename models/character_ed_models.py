import numpy as np
import models


class CharacterModel(object):
    def __init__(self,
                 weights_file,
                 latent_rep_size=None,
                 model=None,
                 charlist = None
                 ):
        self._model = model
        self.charlist = charlist
        # below is the shared code
        self.vae = self._model.MoleculeVAE()
        self.MAX_LEN = self._model.MAX_LEN
        self._char_index = {}
        for ix, char in enumerate(self.charlist):
            self._char_index[char] = ix
        self.vae.load(self.charlist,
                      weights_file,
                      max_length=self.MAX_LEN,
                      latent_rep_size=latent_rep_size)

    def encode(self, smiles):
        """ Encode a list of smiles strings into the latent space """
        indices = [np.array([self._char_index[c] for c in entry], dtype=int) for entry in smiles]
        one_hot = np.zeros((len(indices), self.MAX_LEN, len(self.charlist)), dtype=np.float32)
        for i in range(len(indices)):
            num_productions = len(indices[i])
            one_hot[i][np.arange(num_productions),indices[i]] = 1.
            one_hot[i][np.arange(num_productions, self.MAX_LEN),-1] = 1.
        return self.vae.encoderMV.predict(one_hot)[0]

    def decode(self, z):
        """ Sample from the character decoder """
        assert z.ndim == 2
        out = self.vae.decoder.predict(z)
        noise = np.random.gumbel(size=out.shape)
        sampled_chars = np.argmax(np.log(out) + noise, axis=-1)
        char_matrix = np.array(self.charlist)[np.array(sampled_chars, dtype=int)]
        return [''.join(ch).strip() for ch in char_matrix]


eq_charlist = ['x', '+', '(', ')', '1', '2', '3', '*', '/', 's', 'i', 'n', 'e', 'p', ' ']


class EquationCharacterModel(CharacterModel):
    def __init__(self,
                 weights_file,
                 latent_rep_size=25,
                 model=models.model_eq_str,
                 charlist=eq_charlist
                 ):
        super().__init__(weights_file,
                         latent_rep_size=latent_rep_size,
                         model=model,
                         charlist=charlist
                         )


zinc_charlist = ['C', '(', ')', 'c', '1', '2', 'o', '=', 'O', 'N', '3', 'F', '[',
                         '@', 'H', ']', 'n', '-', '#', 'S', 'l', '+', 's', 'B', 'r', '/',
                         '4', '\\', '5', '6', '7', 'I', 'P', '8', ' ']


class ZincCharacterModel(CharacterModel):
    def __init__(self,
                 weights_file,
                 latent_rep_size=56,
                 model=models.model_zinc_str,
                 charlist = zinc_charlist
                 ):

        super().__init__(weights_file,
                         latent_rep_size=latent_rep_size,
                         model = model,
                         charlist = charlist
                         )


