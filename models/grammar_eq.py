import nltk
import numpy as np
import six
import pdb

gram = """S -> S '+' T
S -> S '*' T
S -> S '/' T
S -> T
T -> '(' S ')'
T -> 'sin(' S ')'
T -> 'exp(' S ')'
T -> 'x'
T -> '1'
T -> '2'
T -> '3'
Nothing -> None"""

# form the CFG and get the start symbol
GCFG = nltk.CFG.fromstring(gram)
start_index = GCFG.productions()[0].lhs()

# collect all lhs symbols, and the unique set of them
all_lhs = [a.lhs().symbol() for a in GCFG.productions()]
lhs_list = []
for a in all_lhs:
    if a not in lhs_list:
        lhs_list.append(a)

D = len(GCFG.productions())

rhs_map = [None] * D
count = 0
for a in GCFG.productions():
    rhs_map[count] = []
    for b in a.rhs():
        if not isinstance(b, six.string_types):
            s = b.symbol()
            rhs_map[count].extend(list(np.where(np.array(lhs_list) == s)[0]))
    count = count + 1

masks = np.zeros((len(lhs_list), D))
count = 0
# all_lhs.append(0)
for sym in lhs_list:
    is_in = np.array([a == sym for a in all_lhs], dtype=int).reshape(1, -1)
    # pdb.set_trace()
    masks[count] = is_in
    count = count + 1

index_array = []
for i in range(masks.shape[1]):
    index_array.append(np.where(masks[:, i] == 1)[0][0])
ind_of_ind = np.array(index_array)

ind_to_lhs_ind = -np.ones(len(all_lhs), dtype=int)
for i,a in enumerate(all_lhs):
    for ind,un_a in enumerate(lhs_list):
        if a==un_a:
            ind_to_lhs_ind[i] = ind

ind_to_lhs_ind = -np.ones(len(all_lhs), dtype=int)
for i, a in enumerate(all_lhs):
    for ind, un_a in enumerate(lhs_list):
        if a == un_a:
            ind_to_lhs_ind[i] = ind
