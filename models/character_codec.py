import numpy as np

class CharacterModel(object):
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
        self.vae = model
        self.vae.eval()

    def string_to_one_hot(self, smiles):
        """ Encode a list of smiles strings into the latent space """
        indices = [np.array([self._char_index[c] for c in entry], dtype=int) for entry in smiles]
        one_hot = np.zeros((len(indices), self.MAX_LEN, len(self.charlist)), dtype=np.float32)
        for i in range(len(indices)):
            num_productions = len(indices[i])
            one_hot[i][np.arange(num_productions),indices[i]] = 1.
            one_hot[i][np.arange(num_productions, self.MAX_LEN),-1] = 1.
        return one_hot

    def encode(self, smiles):
        one_hot = self.string_to_one_hot(smiles)
        z_mean = self.vae.encoder.encode(one_hot)
        return z_mean

    def decode(self, z, validate = False, max_attempts = 10):
        """ Sample from the character decoder """
        assert z.ndim == 2
        out = self.vae.decoder.decode(z)
        if not validate:
            return self.decode_from_onehot(out)
        else:
            import rdkit
            out = []
            for x in out:
                for _ in range(max_attempts):
                    smiles = self.decode_from_onehot(np.array([x]))[0]
                    result = rdkit.Chem.MolFromSmiles(smiles)
                    if result is not None:
                        break
                out.append(smiles)
            return out

    def decode_from_onehot(self, out):
        noise = np.random.gumbel(size=out.shape)
        sampled_chars = np.argmax(np.log(out) + noise, axis=-1)
        char_matrix = np.array(self.charlist)[np.array(sampled_chars, dtype=int)]
        return [''.join(ch).strip() for ch in char_matrix]



# class EquationCharacterModel(CharacterModel):
#     def __init__(self,
#                  weights_file = None,
#                  model=None,
#                  latent_rep_size=None,
#                  max_len = None,
#                  charlist=None,
#                  model_type=models_torch.GrammarVariationalAutoEncoder  # models.model_eq.MoleculeVAE(),
#                  ):
#         settings = get_settings(molecules=False, grammar=False)
#         if latent_rep_size is None:
#             latent_rep_size = settings['z_size']
#         if max_len is None:
#             max_len = settings['max_seq_length']
#         if charlist is None:
#             charlist = settings['charlist']
#
#         super().__init__(max_len=max_len,
#                          charlist=charlist,
#                          model=model,)
#
#
# class ZincCharacterModel(CharacterModel):
#     def __init__(self,
#                  weights_file=None,
#                  model=None,
#                  latent_rep_size=None,
#                  max_len = None,
#                  charlist = None,
#                  model_type = models_torch.GrammarVariationalAutoEncoder
#                  ):
#         settings = get_settings(molecules=True,grammar=False)
#         if latent_rep_size is None:
#             latent_rep_size = settings['z_size']
#         if max_len is None:
#             max_len = settings['max_seq_length']
#         if charlist is None:
#             charlist = settings['charlist']
#
#         super().__init__(weights_file,
#                          latent_rep_size=latent_rep_size,
#                          max_len=max_len,
#                          charlist=charlist,
#                          model=model,
#                          model_type=model_type
#                          )


