import re
import nltk
import numpy as np

import grammarVAE_pytorch.models.grammar_helper as grammar_helper
from grammarVAE_pytorch.models.grammar_mask_gen import GrammarMaskGenerator
from grammarVAE_pytorch.models.reinforcement.reinforcement import SimpleDiscreteDecoder, OneStepDecoderContinuous
from grammarVAE_pytorch.models.policy import SoftmaxRandomSamplePolicy
from grammarVAE_pytorch.models.codec import GenericCodec


class GrammarModel(GenericCodec):
    def __init__(self,
                 model = None,
                 max_len = None,
                 grammar = None,
                 tokenizer = None):
        """ Load the (trained) zinc encoder/decoder, grammar model. """
        self.grammar = grammar
        #self._model = model
        self._tokenize = tokenizer
        self.MAX_LEN = max_len
        self._productions = self.grammar.GCFG.productions()
        self._prod_map = {}
        for ix, prod in enumerate(self._productions):
            self._prod_map[prod] = ix
        self._parser = nltk.ChartParser(self.grammar.GCFG)
        self._n_chars = len(self._productions)

        if model is not None:
            self.vae = model
            self.vae.eval()
            self.decoder = self.vae.decoder
        # self.mask_gen = GrammarMaskGenerator(self.MAX_LEN, grammar=self.grammar)
        # policy = SoftmaxRandomSamplePolicy()
        # stepper = OneStepDecoderContinuous(self.vae.decoder)
        # self.decoder = SimpleDiscreteDecoder(stepper, policy, self.mask_gen, bypass_actions = True)


    def string_to_one_hot(self, smiles):
        """ Encode a list of smiles strings into the latent space """
        assert type(smiles) == list
        tokens = map(self._tokenize, smiles)
        parse_trees = [next(self._parser.parse(t)) for t in tokens]
        productions_seq = [tree.productions() for tree in parse_trees]
        indices = [np.array([self._prod_map[prod] for prod in entry], dtype=int) for entry in productions_seq]
        one_hot = np.zeros((len(indices), self.MAX_LEN, self._n_chars), dtype=np.float32)
        for i in range(len(indices)):
            num_productions = len(indices[i])
            one_hot[i][np.arange(num_productions), indices[i]] = 1.
            one_hot[i][np.arange(num_productions, self.MAX_LEN), -1] = 1.
        #self.one_hot = one_hot
        return one_hot

    # def string_to_actions(self,smiles):
    #     """ Encode a list of smiles strings into the latent space """
    #     assert type(smiles) == list
    #     tokens = map(self._tokenize, smiles)
    #     parse_trees = [next(self._parser.parse(t)) for t in tokens]
    #     productions_seq = [tree.productions() for tree in parse_trees]
    #     indices = [np.array([self._prod_map[prod] for prod in entry], dtype=int) for entry in productions_seq]
    # # TODO: move to superclass?

    def decode_from_actions(self, actions):
        '''
        Takes a batch of action sequences, applies grammar
        :param actions: batch_size x seq_length LongTensor or ndarray(ints)
        :return: list of strings
        '''
        # Convert from one-hot to sequence of production rules
        prod_seq = [[self._productions[actions[index,t]]
                     for t in range(actions.shape[1])]
                    for index in range(actions.shape[0])]
        out = []
        for ip, prods in enumerate(prod_seq):
            out.append(prods_to_eq(prods))

        return out


def eq_tokenizer(s):
    funcs = ['sin', 'exp']
    for fn in funcs: s = s.replace(fn+'(', fn+' ')
    s = re.sub(r'([^a-z ])', r' \1 ', s)
    for fn in funcs: s = s.replace(fn, fn+'(')
    return s.split()


def get_zinc_tokenizer(cfg):
    long_tokens = [a for a in cfg._lexical_index.keys() if  len(a) > 1 ] #filter(lambda a: len(a) > 1, cfg._lexical_index.keys())
    replacements = ['$','%','^'] # ,'&']
    assert len(long_tokens) == len(replacements)
    for token in replacements:
        #assert not cfg._lexical_index.has_key(token)
        assert not token in cfg._lexical_index

    def tokenize(smiles):
        for i, token in enumerate(long_tokens):
            smiles = smiles.replace(token, replacements[i])
        tokens = []
        for token in smiles:
            try:
                ix = replacements.index(token)
                tokens.append(long_tokens[ix])
            except:
                tokens.append(token)
        return tokens

    return tokenize

zinc_tokenizer = get_zinc_tokenizer(grammar_helper.grammar_zinc.GCFG)


def prods_to_eq(prods):
    seq = [prods[0].lhs()]
    for prod in prods:
        if str(prod.lhs()) == nltk.grammar.Nonterminal('Nothing'):
            break
        for ix, s in enumerate(seq):
            if s == prod.lhs():
                seq = seq[:ix] + list(prod.rhs()) + seq[ix+1:]
                break
    try:
        return ''.join(seq)
    except:
        raise Exception("We've run out of max_length but still have nonterminals: something is wrong here...")



