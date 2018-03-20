# import sys
# sys.path.append('..')
import os, inspect
from grammarVAE_pytorch.models import grammar_ed_models as grammar_model
import numpy as np

# We load the auto-encoder
my_location = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
grammar_model = grammar_model.ZincGrammarModel()
z = grammar_model.encode(['c1nccc2n1ccc2'])
for _ in range(100):
    z = np.random.normal(size=(1,56))
    new_smile = grammar_model.decode(z)
    print(new_smile)