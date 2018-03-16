import re
import nltk
import numpy as np



import grammarVAE_pytorch.models.model_grammar_pytorch as models_torch
import grammarVAE_pytorch.models.grammar_helper as grammar_helper


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
        self._lhs_map = {}
        for ix, lhs in enumerate(self.grammar.lhs_list):
            self._lhs_map[lhs] = ix
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

    def smiles_to_one_hot(self,smiles):
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
        one_hot = self.smiles_to_one_hot(smiles)
        z_mean = self.vae.encoder.encode(one_hot)
        return z_mean

    def _sample_using_masks(self, unmasked):
        """ Samples a one-hot vector, masking at each timestep.
            This is an implementation of Algorithm ? in the paper. """
        eps = 1e-100
        X_hat = np.zeros_like(unmasked)

        # Create a stack for each input in the batch
        S = np.empty((unmasked.shape[0],), dtype=object)
        for ix in range(S.shape[0]):
            S[ix] = [self.grammar.start_index]

        # Loop over time axis, sampling values and updating masks
        tdist_reduction = [False for _ in range(len(S))]
        for t in range(unmasked.shape[1]):
            # print('S:', S,[self.grammar.terminal_dist(sym) for sym in S[0]],
            #       sum([self.grammar.terminal_dist(sym) for sym in S[0]]),
            #       self.MAX_LEN - t,self.MAX_LEN - t - self.grammar.max_term_dist_increase - 1 )
            term_distance = [sum([self.grammar.terminal_dist(sym) for sym in s]) for s in S]
            next_nonterminal = [self._lhs_map[pop_or_nothing(a)] for a in S]
            mask = self.grammar.masks[next_nonterminal]

            for ix, s in enumerate(S):
                # if the number of total steps left to all terminals is at least the length of the sequence
                #print('s:',s,[self.grammar.terminal_dist(sym) for sym in s],self.MAX_LEN - t - self.grammar.max_term_dist_increase - 3)
                if term_distance[ix] >= self.MAX_LEN - t - self.grammar.max_term_dist_increase - 1:
                    tdist_reduction[ix] = True # go into terminal distance reduction mode for that molecule
                if tdist_reduction[ix]:
                    mask[ix] *= self.grammar.terminal_mask[0]
                    #print('****')
            masked_output = np.exp(unmasked[:,t,:])*mask + eps
            sampled_output = np.argmax(np.random.gumbel(size=masked_output.shape) + np.log(masked_output), axis=-1)
            this_prod = self.grammar.GCFG.productions()[sampled_output[0]]
            #print(this_prod, self.grammar.terminal_dist(this_prod.lhs()),[self.grammar.terminal_dist(x) for x in this_prod.rhs()], tdist_reduction[ix])
            X_hat[np.arange(unmasked.shape[0]),t,sampled_output] = 1.0

            # Identify non-terminals in RHS of selected production, and
            # push them onto the stack in reverse order
            # rhs = [filter(lambda a: (type(a) == nltk.grammar.Nonterminal) and (str(a) != 'None'),
            #               self._productions[i].rhs()) for i in sampled_output]

            rhs =[[x for x in self._productions[sampled_ind].rhs()
                   if (type(x) == nltk.grammar.Nonterminal) and (str(x) != 'None')]
                        for sampled_ind in sampled_output]


            #rhs = [type(i) for i in sampled_output if (str(i) != 'None')]

            for ix, this_rhs in enumerate(rhs):
                S[ix].extend([x for x in this_rhs[::-1]])
        return X_hat # , ln_p

    # TODO: keep trying to decode for max_attempts, until rdkit likes it!
    def decode(self, z, validate = False, max_attempts = 10):
        """ Sample from the grammar decoder """
        assert z.ndim == 2
        unmasked = self.vae.decoder.decode(z)
        if not validate:
            return self.decode_from_onehot(unmasked)
        else:
            import rdkit
            out = []
            for x in unmasked:
                for _ in range(max_attempts):
                    smiles = self.decode_from_onehot(np.array([x]))[0]
                    result = rdkit.Chem.MolFromSmiles(smiles)
                    if result is not None:
                        break
                out.append(smiles)
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

        return out



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
    try:
        return S.pop()
    except:
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
        return ''

try:
    # inside a try-catch block so the rest also runs without rdkit
    from rdkit.Chem import MolFromSmiles

    def fraction_valid(smiles):
        valid = [s for s in smiles if s != '' and MolFromSmiles(s) is not None]
        invalid = [s for s in smiles if s != '' and MolFromSmiles(s) is None]
        valid_lens = [len(MolFromSmiles(s).GetAtoms()) for s in valid]
        num_valid = len(valid_lens)
        avg_len = sum(valid_lens) / (len(valid_lens) + 1e-6)
        max_len = 0 if not len(valid_lens) else max(valid_lens)
        print(valid)
        return (num_valid, avg_len, max_len), (valid,invalid)
except:
    pass