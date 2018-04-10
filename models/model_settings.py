import os, inspect

from grammarVAE_pytorch.models import variational_autoencoder as models_torch
from grammarVAE_pytorch.models.character_codec import CharacterModel
from grammarVAE_pytorch.models.grammar_codec import GrammarModel, zinc_tokenizer, eq_tokenizer
from grammarVAE_pytorch.models.grammar_helper import grammar_eq, grammar_zinc
from grammarVAE_pytorch.models.grammar_mask_gen import GrammarMaskGenerator
from basic_pytorch.models.rnn_models import SimpleRNNDecoder, SimpleRNNAttentionEncoder, ResettingRNNDecoder
from grammarVAE_pytorch.models.encoders import SimpleCNNEncoder
from grammarVAE_pytorch.models.decoders import OneStepDecoder, OneStepDecoderUsingAction, OneStepDecoderContinuous, \
    SimpleDiscreteDecoder
from grammarVAE_pytorch.models.policy import SoftmaxRandomSamplePolicy
from basic_pytorch.gpu_utils import to_gpu
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
                        'decoder_hidden_n': 501, #mkusner/grammarVAE has 501 but that eats too much GPU :)
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
            #from grammarVAE_pytorch.models.character_ed_models import ZincCharacterModel as ThisModel
            settings = {'source_data': root_location + 'data/250k_rndm_zinc_drugs_clean.smi',
                        'data_path': root_location + 'data/zinc_str_dataset.h5',
                        'filename_stub': 'char_zinc_',
                        'charlist': zinc_charlist,
                        'grammar': None,
                        'z_size': 292,
                        'decoder_hidden_n': 501, #mkusner/grammarVAE has 501 but that eats too much GPU :)
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


def get_model(molecules=True, grammar = True, weights_file=None, epsilon_std=1, **kwargs):
    model_args = get_model_args(molecules=molecules, grammar=grammar)
    for key, value in kwargs.items():
        if key in model_args:
            model_args[key] = value
    sample_z = model_args.pop('sample_z')
    encoder, decoder = get_encoder_decoder(molecules,grammar,**model_args)
    model = models_torch.GrammarVariationalAutoEncoder(encoder=encoder,
                                                       decoder=decoder,
                                                       sample_z=sample_z,
                                                       epsilon_std=epsilon_std)

    if weights_file is not None:
        model.load(weights_file)

    settings = get_settings(molecules=molecules, grammar=grammar)
    if grammar:
        wrapper_model = GrammarModel(max_len=settings['max_seq_length'],
                                 grammar=settings['grammar'],
                                 tokenizer=zinc_tokenizer if molecules else eq_tokenizer,
                                 model=model)
    else:
        wrapper_model=CharacterModel(max_len=settings['max_seq_length'],
                                     charlist=settings['charlist'],
                                     model=model
                                     )
    return model, wrapper_model

def get_encoder_decoder(molecules = True,
                        grammar=True,
                        z_size=200,
                 decoder_hidden_n=200,
                 feature_len=12,
                 max_seq_length=15,
                 cnn_encoder_params={'kernel_sizes': (2, 3, 4),
                                               'filters': (2, 3, 4),
                                               'dense_size': 100},
                 drop_rate = 0.0,
                 rnn_encoder = False,
                 rnn_encoder_hidden_n = 200):
    settings = get_settings(molecules,grammar)
    if rnn_encoder:
        encoder = to_gpu(SimpleRNNAttentionEncoder(max_seq_length=max_seq_length,
                                                        z_size=z_size,
                                                        hidden_n=rnn_encoder_hidden_n,
                                                        feature_len=feature_len,
                                                        drop_rate=drop_rate))
    else:
        encoder = to_gpu(SimpleCNNEncoder(params=cnn_encoder_params,
                                               max_seq_length=max_seq_length,
                                               z_size=z_size,
                                               feature_len=feature_len,
                                               drop_rate=drop_rate))
    decoder_type = 'step'
    if decoder_type=='old':
        pre_decoder = ResettingRNNDecoder(z_size=z_size,
                                           hidden_n=decoder_hidden_n,
                                           feature_len=feature_len,
                                           max_seq_length=max_seq_length,
                                           drop_rate=drop_rate)
        stepper = OneStepDecoderContinuous(pre_decoder)
    elif decoder_type=='step':
        pre_decoder = SimpleRNNDecoder(z_size=z_size,
                                          hidden_n=decoder_hidden_n,
                                          feature_len=feature_len,
                                          max_seq_length=1,
                                          drop_rate=drop_rate)
        stepper = OneStepDecoder(pre_decoder, max_len=max_seq_length)
    elif decoder_type=='action':
        pre_decoder = SimpleRNNDecoder(z_size=z_size + feature_len,
                                       hidden_n=decoder_hidden_n,
                                       feature_len=feature_len,
                                       max_seq_length=1,
                                       drop_rate=drop_rate)
        stepper = OneStepDecoderUsingAction(pre_decoder,
                                            max_len=max_seq_length,
                                            num_actions=feature_len)
    if grammar:
        mask_gen = GrammarMaskGenerator(max_seq_length, grammar=settings['grammar'])
    else:
        mask_gen = None
    policy = SoftmaxRandomSamplePolicy()

    decoder = to_gpu(SimpleDiscreteDecoder(stepper, policy, mask_gen))#, bypass_actions=True))

    return encoder, decoder

if __name__=='__main__':
    for molecules in [True, False]:
        for grammar in [True,False]:
            wrapper_model = get_model(molecules=molecules, grammar=grammar)

    print('success!')