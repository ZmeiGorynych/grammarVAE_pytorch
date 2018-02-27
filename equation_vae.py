import nltk
import re

import eq_grammar
from models.grammar_model import GrammarModel
import molecule_vae
import models.model_eq
import models.model_eq_str
import numpy as np

def tokenize(s):
    funcs = ['sin', 'exp']
    for fn in funcs: s = s.replace(fn+'(', fn+' ')
    s = re.sub(r'([^a-z ])', r' \1 ', s)
    for fn in funcs: s = s.replace(fn, fn+'(')
    return s.split()

class EquationGrammarModel(GrammarModel):
    def __init__(self, weights_file, latent_rep_size=56,
                 grammar=None,
                 model=None,
                 tokenizer=None):
        """ Load the (trained) zinc encoder/decoder, grammar model. """
        if not grammar:
            grammar = eq_grammar
        if not model:
            model = models.model_eq
        if not tokenizer:
            tokenizer = tokenize

        super().__init__(weights_file,
                         latent_rep_size=latent_rep_size,
                         grammar=grammar,
                         model=model,
                         tokenizer=tokenizer)


class EquationCharacterModel(object):

    def __init__(self, weights_file, latent_rep_size=25):
        self._model = models.model_eq_str
        self.MAX_LEN = 19
        self.vae = self._model.MoleculeVAE()
        self.charlist = ['x', '+', '(', ')', '1', '2', '3', '*', '/', 's', 'i', 'n', 'e', 'p', ' ']
        #self.charlist = ['C', '(', ')', 'c', '1', '2', 'o', '=', 'O', 'N', '3', 'F', '[',
        #                 '@', 'H', ']', 'n', '-', '#', 'S', 'l', '+', 's', 'B', 'r', '/',
        #                 '4', '\\', '5', '6', '7', 'I', 'P', '8', ' ']
        self._char_index = {}
        for ix, char in enumerate(self.charlist):
            self._char_index[char] = ix
        self.vae.load(self.charlist, weights_file, max_length=self.MAX_LEN, latent_rep_size=latent_rep_size)

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
