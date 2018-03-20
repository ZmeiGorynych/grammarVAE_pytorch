import os, inspect
from grammarVAE_pytorch.models.grammar_helper import grammar_eq, grammar_zinc
# in the desired end state, this file will contain every single difference between the different models

root_location = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
root_location = root_location + '/../'

eq_charlist = ['x', '+', '(', ')', '1', '2', '3', '*', '/', 's', 'i', 'n', 'e', 'p', ' ']
zinc_charlist =  ['C', '(', ')', 'c', '1', '2', 'o', '=', 'O', 'N', '3', 'F', '[',
                             '@', 'H', ']', 'n', '-', '#', 'S', 'l', '+', 's', 'B', 'r', '/',
                             '4', '\\', '5', '6', '7', 'I', 'P', '8', ' ']

def get_settings(molecules = True, grammar = True):
    if molecules:
        if grammar:
            settings = {'source_data': root_location + 'data/250k_rndm_zinc_drugs_clean.smi',
                        'data_path':root_location + 'data/zinc_grammar_dataset.h5',
                        'filename_stub': 'gramar_zinc_',
                        'grammar': grammar_zinc,
                        'z_size': 56,
                        'decoder_hidden_n': 501,
                        'feature_len': len(grammar_zinc.GCFG.productions()),
                        'max_seq_length': 277,
                        'cnn_encoder_params':{'kernel_sizes': (9, 9, 11),
                                              'filters': (9, 9, 10),
                                              'dense_size': 435},
                        'rnn_encoder_hidden_n': 200,
                        'EPOCHS': 100,
                        'BATCH_SIZE': 300
                        }
        else:
            from grammarVAE_pytorch.models.character_ed_models import ZincCharacterModel as ThisModel
            settings = {'source_data': root_location + 'data/250k_rndm_zinc_drugs_clean.smi',
                        'data_path': root_location + 'data/zinc_str_dataset.h5',
                        'filename_stub': 'char_zinc_',
                        'charlist': zinc_charlist,
                        'grammar': None,
                        'z_size': 292,
                        'decoder_hidden_n': 501,
                        'feature_len': len(zinc_charlist),
                        'max_seq_length': 120,
                        'cnn_encoder_params':{'kernel_sizes': (9, 9, 11),
                                              'filters': (9, 9, 10),
                                              'dense_size': 435},

                        'rnn_encoder_hidden_n': 200,
                        'EPOCHS': 100,
                        'BATCH_SIZE': 500
                        }
    else:
        if grammar:
            settings = {'source_data': root_location + 'data/equation2_15_dataset.txt',
                        'data_path': root_location + 'data/eq2_grammar_dataset.h5',
                        'filename_stub': 'grammar_eq_',
                        'grammar': grammar_eq,
                        'z_size': 25,
                        'decoder_hidden_n': 100,
                        'feature_len': len(grammar_eq.GCFG.productions()),
                        'max_seq_length': 15,
                        'cnn_encoder_params':{'kernel_sizes': (2, 3, 4),
                                              'filters': (2, 3, 4),
                                              'dense_size': 100},
                        'rnn_encoder_hidden_n': 100,
                        'EPOCHS': 50,
                        'BATCH_SIZE':600
                        }
        else:
            settings = {'source_data': root_location + 'data/equation2_15_dataset.txt',
                        'data_path': root_location + 'data/eq2_str_dataset.h5',
                        'filename_stub': 'char_eq_',
                        'charlist': eq_charlist,
                        'grammar': None,
                        'z_size': 25,
                        'decoder_hidden_n': 100,
                        'feature_len': len(eq_charlist),
                        'max_seq_length': 31,# max([len(l) for l in L]) L loaded from textfile
                        'cnn_encoder_params': {'kernel_sizes': (2, 3, 4),
                                               'filters': (2, 3, 4),
                                               'dense_size': 100},
                        'rnn_encoder_hidden_n': 100,
                        'EPOCHS': 50,
                        'BATCH_SIZE': 600
                        }

    return settings

def get_model_args(molecules, grammar,
                   drop_rate=0.5,
                   sample_z = False,
                   rnn_encoder =True):

    settings = get_settings(molecules,grammar)
    model_args = {'z_size': settings['z_size'],
                  'decoder_hidden_n':  settings['decoder_hidden_n'],
                  'feature_len': settings['feature_len'],
                  'max_seq_length': settings['max_seq_length'],
                  'cnn_encoder_params':  settings['cnn_encoder_params'],
                  'drop_rate': drop_rate,
                  'sample_z': sample_z,
                  'rnn_encoder': rnn_encoder,
                  'rnn_encoder_hidden_n': settings['rnn_encoder_hidden_n']}

    return model_args