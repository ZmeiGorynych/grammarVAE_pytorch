import re
import nltk
import numpy as np



import grammarVAE_pytorch.models.model_grammar_pytorch as models_torch
import grammarVAE_pytorch.models.grammar_helper as grammar_helper


class GrammarMaskGenerator:
    def __init__(self, S_len, MAX_LEN, grammar=None):
        self.S = [[] for _ in range(S_len)]
        self.tdist_reduction = [False for _ in range(len(self.S))]
        self.t = 0
        self.MAX_LEN = MAX_LEN
        self.grammar = grammar
        self._lhs_map ={lhs: ix for ix, lhs in enumerate(self.grammar.lhs_list)}

    def next_mask(self, last_action):
        '''
        Consumes one action at a time, responds with the mask for next action
        : param last_action: previous action, array of ints of len = batch_size; None for the very first step
        '''
        if last_action is None:
            # this is how we start
            self.S = [[self.grammar.start_index] for _ in range(len(self.S))]

        else:
            # insert the non-terminals from last action into the stack in reverse order
            rhs = [[x for x in self.grammar.GCFG.productions()[sampled_ind].rhs()
                    if (type(x) == nltk.grammar.Nonterminal) and (str(x) != 'None')]
                        for sampled_ind in last_action]

            for ix, this_rhs in enumerate(rhs):
                self.S[ix] += [x for x in this_rhs[::-1]]

        # Have to calculate total terminal distance BEFORE we pop the next nonterminal!
        self.term_distance = [sum([self.grammar.terminal_dist(sym) for sym in s]) for s in self.S]

        # get the next nonterminal and look up the mask for it
        next_nonterminal = [self._lhs_map[pop_or_nothing(a)] for a in self.S]
        mask = self.grammar.masks[next_nonterminal]

        # add masking to make sure the sequence always completes
        # TODO: vectorize this
        for ix, s in enumerate(self.S):
            #term_distance = sum([self.grammar.terminal_dist(sym) for sym in s])
            if self.term_distance[ix] >= self.MAX_LEN - self.t - self.grammar.max_term_dist_increase - 1:
                self.tdist_reduction[ix] = True  # go into terminal distance reduction mode for that molecule
            if self.tdist_reduction[ix]:
                mask[ix] *= self.grammar.terminal_mask[0]


        self.t += 1
        return mask

class GrammarModel(object):
    def __init__(self,
                 weights_file =None,
                 model = None,
                 rnn_encoder=True,
                 latent_rep_size=None,
                 max_len = None,
                 grammar = None,
                 model_type=models_torch.GrammarVariationalAutoEncoder,
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
        # self._lhs_map = {}
        # for ix, lhs in enumerate(self.grammar.lhs_list):
        #     self._lhs_map[lhs] = ix
        if model is not None:
            self.vae = model
        else:
            # assume model hidden_n and encoder_kernel_size are always the same
            # TODO: should make better use of model_settings here!
            self.vae = model_type(z_size=latent_rep_size,
                             feature_len=len(self._productions),
                             max_seq_length=self.MAX_LEN,
                             rnn_encoder=rnn_encoder)
            if weights_file is not None:
                self.vae.load(weights_file)
        self.vae.eval()

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

    def encode(self, smiles):
        one_hot = self.string_to_one_hot(smiles)
        z_mean = self.vae.encoder.encode(one_hot)
        return z_mean

    def _sample_using_masks(self, unmasked):
        """ Samples a one-hot vector, masking at each timestep.
            This is an implementation of Algorithm ? in the paper. """
        eps = 1e-100
        X_hat = np.zeros_like(unmasked)
        mask_gen = GrammarMaskGenerator(unmasked.shape[0], self.MAX_LEN, grammar=self.grammar)
        # Loop over time axis, sampling values and updating masks
        sampled_output = None
        for t in range(unmasked.shape[1]):
            mask = mask_gen.next_mask(sampled_output)
            masked_output = np.exp(unmasked[:,t,:])*mask + eps
            sampled_output = np.argmax(np.random.gumbel(size=masked_output.shape) + np.log(masked_output), axis=-1)
            X_hat[np.arange(unmasked.shape[0]),t,sampled_output] = 1.0
        return X_hat # , ln_p

    # TODO: keep trying to decode for max_attempts, until rdkit likes it!
    def decode_ext(self, z, validate = False, max_attempts = 10):
        """ Sample from the grammar decoder """
        assert z.ndim == 2
        unmasked = self.vae.decoder.decode(z)
        if not validate:
            out, X_hat = self.decode_from_onehot(unmasked)
            return out, X_hat
        else:
            import rdkit
            out = []
            X_hats = []
            for x in unmasked:
                for _ in range(max_attempts):
                    smiles, X_hat = self.decode_from_onehot(np.array([x]))
                    result = rdkit.Chem.MolFromSmiles(smiles[0])
                    if result is not None:
                        break
                out.append(smiles[0])
                X_hats.append(X_hat)
            return out, np.concatenate(X_hats, axis=0)

    def decode(self, z, validate = False, max_attempts = 10):
        out, _ = self.decode_ext(z, validate, max_attempts)
        return out

    def decode_from_onehot(self, unmasked):
        X_hat = self._sample_using_masks(unmasked)
        # Convert from one-hot to sequence of production rules
        prod_seq = [[self._productions[X_hat[index,t].argmax()]
                     for t in range(X_hat.shape[1])]
                    for index in range(X_hat.shape[0])]
        out = []
        for ip, prods in enumerate(prod_seq):
            out.append(prods_to_eq(prods))

        return out, X_hat



def eq_tokenizer(s):
    funcs = ['sin', 'exp']
    for fn in funcs: s = s.replace(fn+'(', fn+' ')
    s = re.sub(r'([^a-z ])', r' \1 ', s)
    for fn in funcs: s = s.replace(fn, fn+'(')
    return s.split()


class EquationGrammarModel(GrammarModel):
    def __init__(self, weights_file = None,
                 model=None,
                 latent_rep_size=56,
                 max_len=15,
                 grammar=grammar_helper.grammar_eq,
                 model_type=models_torch.GrammarVariationalAutoEncoder,#models.model_eq.MoleculeVAE(),
                 tokenizer=eq_tokenizer):
        """ Load the (trained) zinc encoder/decoder, grammar model. """
        super().__init__(weights_file,
                         latent_rep_size=latent_rep_size,
                         max_len=max_len,
                         grammar=grammar,
                         model=model,
                         model_type=model_type,
                         tokenizer=tokenizer)

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


# TODO: get all the zinc vs eq bits from model_settings!
zinc_tokenizer = get_zinc_tokenizer(grammar_helper.grammar_zinc.GCFG)


class ZincGrammarModel(GrammarModel):
    def __init__(self,
                 weights_file=None,
                 model=None,
                 rnn_encoder=True,
                 latent_rep_size=56,
                 max_len=277,
                 grammar=grammar_helper.grammar_zinc,
                 model_type =models_torch.GrammarVariationalAutoEncoder,#models.model_zinc.MoleculeVAE(),
                 tokenizer=zinc_tokenizer):
        super().__init__(weights_file,
                         rnn_encoder=rnn_encoder,
                         latent_rep_size=latent_rep_size,
                         max_len=max_len,
                         grammar=grammar,
                         model=model,
                         model_type=model_type,
                         tokenizer=tokenizer)


def pop_or_nothing(S):
    if len(S):
        return S.pop()
    else:
        return nltk.grammar.Nonterminal('Nothing')


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

