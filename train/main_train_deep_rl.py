import math
from deep_rl import *
from grammarVAE_pytorch.models.rdkit_utils import num_atoms
from models.problem.rl.DeepRL_wrappers import BodyAdapter, MyA2CAgent
from models.problem.rl.env_and_task import SequenceGenerationTask
import logging

def reward_length(smiles):
    '''
    A simple reward to encourage larger molecule length
    :param smiles: list of strings
    :return: reward, list of float
    '''
    atoms = num_atoms(smiles)
    return [-1 if num is None else math.sqrt(num) for num in atoms]

batch_size = 10

def a2c_sequence(name = 'a2c_sequence', task=None, body=None):
    config = Config()
    config.num_workers = batch_size # same thing as batch size
    config.task_fn = lambda: task
    config.optimizer_fn = lambda params: torch.optim.RMSprop(params, lr=0.0007)
    config.network_fn = lambda state_dim, action_dim: CategoricalActorCriticNet(
        state_dim, action_dim, body, gpu=0)
    config.policy_fn = SamplePolicy
    config.state_normalizer = lambda x: x
    config.reward_normalizer = lambda x: x
    config.discount = 0.99
    config.use_gae = False #TODO: for now, MUST be false as our RNN network isn't com
    config.gae_tau = 0.97
    config.entropy_weight = 0.01
    config.rollout_length = 5
    config.gradient_clip = 0.5
    config.logger = logging.getLogger()#get_logger(file_name='deep_rl_a2c', skip=True)
    run_iterations(MyA2CAgent(config))


drop_rate = 0.2
molecules = True
grammar = False

task = SequenceGenerationTask(molecules = molecules,
                              grammar = grammar,
                              reward_fun = reward_length,
                              batch_size = batch_size)
#
# from transformer.OneStepAttentionDecoder import SelfAttentionDecoderStep
# decoder = SelfAttentionDecoderStep(num_actions=task.env.action_dim,
#                                        max_seq_len=task.env._max_episode_steps,
#                                        drop_rate=drop_rate)

from generative_playground.models.decoder.basic_rnn import SimpleRNNDecoder
decoder = SimpleRNNDecoder(z_size=5,
                               hidden_n=64,
                               feature_len=task.env.action_dim,
                               max_seq_length=task.env._max_episode_steps,  # TODO: WHY???
                               drop_rate=drop_rate,
                               use_last_action=False)

body = BodyAdapter(decoder)

a2c_sequence(task=task, body=body)