# import sys
# sys.path.append('..')
import os, inspect
from grammarVAE_pytorch.models import grammar_ed_models as grammar_model

# We load the auto-encoder
my_location = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
grammar_weights = my_location + '/pretrained/my_molecules.mdl'
grammar_model = grammar_model.ZincGrammarModel(grammar_weights)
z = grammar_model.encode(['c1nccc2n1ccc2'])
print(type(z))
new_smile = grammar_model.decode(z)
print(new_smile)