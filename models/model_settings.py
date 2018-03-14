import os, inspect
root_location = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
root_location = root_location + '/../'
from grammarVAE_pytorch.models.grammar_helper import grammar_eq, grammar_zinc

# These are settings that are meant to be pretty much constant
settings_zinc = {'grammar':grammar_zinc,
                 'data_path':root_location + 'data/zinc_grammar_dataset.h5',
                 'z_size': 56,
                  'hidden_n': 200,
                  'feature_len': len(grammar_zinc.GCFG.productions()),
                  'max_seq_length': 277,
                  'encoder_kernel_sizes': (2, 3, 4),
                  'EPOCHS': 100,
                  'BATCH_SIZE': 300
                  }

settings_eq = {'grammar':grammar_eq,
               'data_path': root_location + 'data/eq2_grammar_dataset.h5',
                'z_size': 25,
              'hidden_n': 200,
              'feature_len': len(grammar_eq.GCFG.productions()),
              'max_seq_length': 15,
              'encoder_kernel_sizes': (2, 3, 4),
               'EPOCHS': 50,# from mkusner/grammarVAE
               'BATCH_SIZE':600# from mkusner/grammarVAE
                }

def get_model_args(molecules,
                   drop_rate=0.5,
                   sample_z = False,
                   rnn_encoder =True):
    if molecules:
        settings = settings_zinc
    else:
        settings = settings_eq

    model_args = {'z_size': settings['z_size'],
                  'hidden_n':  settings['hidden_n'],
                  'feature_len': settings['feature_len'],
                  'max_seq_length': settings['max_seq_length'],
                  'encoder_kernel_sizes':  settings['encoder_kernel_sizes'],
                  'drop_rate': drop_rate,
                  'sample_z': sample_z,
                  'rnn_encoder': rnn_encoder}

    return model_args