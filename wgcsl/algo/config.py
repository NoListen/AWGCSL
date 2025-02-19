import os
import numpy as np
import gym

from wgcsl.common import logger
from wgcsl.algo.wgcsl import WGCSL
from wgcsl.algo.supervised_sampler import make_sample_transitions, make_random_sample
from wgcsl.common.monitor import Monitor
from wgcsl.envs.multi_world_wrapper import PointGoalWrapper, SawyerGoalWrapper, ReacherGoalWrapper, FetchGoalWrapper

# offline parameters
DEFAULT_ENV_PARAMS = {
    'SawyerReachXYZEnv-v1':{
        'n_cycles':10,
        'baw_delta': 0.15,
    },
    'SawyerPushAndReachEnvEasy-v0':{
        'n_cycles':10,
        'n_batches': 40,   
        'baw_delta': 0.15,
    },
    'FetchSlide-v1':{
        'batch_size': 1024, 
        'n_cycles': 50,  
        'n_batches': 40, 
        'baw_delta': 0.01,
        },
    'FetchPickAndPlace-v1':{
        'batch_size': 1024,
        'n_cycles': 50,
        'n_batches': 40,
        'baw_delta': 0.01,
        },
    'HandReach-v0':{
        'batch_size': 1024,
        'n_cycles': 50,  
        'n_batches': 40, 
        'baw_delta': 0.01,
        },
    'HandManipulateBlockRotateXYZ-v0':{
        'batch_size': 1024,
        'n_cycles': 50,
        'n_batches': 40,
        'baw_delta': 0.01,
        },
}


DEFAULT_PARAMS = {  
    # env
    'max_u': 1.,  # max absolute value of actions on each coordinate
    'layers': 3,  # number of layers in the critic/actor networks
    'hidden': 256,  # number of neurons in each hidden layers
    'network_class': 'wgcsl.algo.actor_critic:ActorCritic',
    'Q_lr': 5e-4,  # critic learning rate
    'pi_lr': 5e-4,  # actor learning rate
    'buffer_size': int(1E6),  # for experience replay
    'polyak': 0.9,  #polyak averaging coefficient
    'action_l2': 1.0,  # quadratic penalty on actions (before rescaling by max_u)
    'clip_obs': 200.,
    'scope': 'wgcsl',
    'relative_goals': False,
    # training
    'num_epoch':50, 
    'n_cycles': 10,  # per epoch
    'rollout_batch_size': 1,  # per mpi thread
    'n_batches': 4,  # training batches per cycle
    'batch_size': 1024,  # per mpi thread, measured in transitions and reduced to even multiple of chunk_length.
    'n_test_rollouts': 100,  # number of test rollouts per epoch, each consists of rollout_batch_size rollouts
    'test_with_polyak': False,  # run test episodes with the target network
    # exploration, not used in the offline setting
    'random_eps': 0.3,  # percentage of time a random action is taken
    'noise_eps': 0.2,  # std of gaussian noise added to not-completely-random actions as a percentage of max_u
    # random init episode, not used int the offline setting
    'random_init':20,

    # goal relabeling
    'replay_strategy': 'future',  
    'replay_k': 4,  # number of additional goals used for replay
    # normalization
    'norm_eps': 1e-4,  # epsilon used for observation normalization
    'norm_clip': 5,  # normalized observations are cropped to this values
    # use supervised
    'use_supervised': False,
    # best-advantage weight
    'baw_delta': 0.1,
    'baw_max': 80,

    # if do not use her
    'no_relabel':False,    # used for no relabel
    
    'use_huber': False,
    'delta': 1.0,
    'grad_clip_value': -1
}


CACHED_ENVS = {}


def cached_make_env(make_env):
    """
    Only creates a new environment from the provided function if one has not yet already been
    created. This is useful here because we need to infer certain properties of the env, e.g.
    its observation and action spaces, without any intend of actually using it.
    """
    if make_env not in CACHED_ENVS:
        env = make_env()
        CACHED_ENVS[make_env] = env
    return CACHED_ENVS[make_env]

def prepare_mode(kwargs):
    if 'mode' in kwargs.keys():
        mode = kwargs['mode']
        if mode == 'supervised':
            kwargs['use_supervised'] = True
        else:
            kwargs['use_supervised'] = False
    else:
        kwargs['use_supervised'] = False
    return kwargs


def prepare_params(kwargs):
    # default max episode steps
    kwargs = prepare_mode(kwargs)
    default_max_episode_steps = 50
    # WGCSL params
    wgcsl_params = dict()
    env_name = kwargs['env_name']
    def make_env(subrank=None):
        try:
            env = gym.make(env_name, rewrad_type='sparse') 
        except:
            logger.log('Can not make sparse reward environment')
            env = gym.make(env_name)
        # add wrapper for multiworld environment
        if env_name.startswith('Fetch'):
            env._max_episode_steps = 50
            env = FetchGoalWrapper(env)
        elif env_name.startswith('HandManipulate'):
            env._max_episode_steps = 100
        elif env_name.startswith('Point'):
            env = PointGoalWrapper(env)
            env.env._max_episode_steps = 50
        elif env_name.startswith('Sawyer'): 
            env = SawyerGoalWrapper(env)
            env.env._max_episode_steps = 100
        elif env_name.startswith('Reacher'):
            env = ReacherGoalWrapper(env)

        if (subrank is not None and logger.get_dir() is not None):
            try:
                from mpi4py import MPI
                mpi_rank = MPI.COMM_WORLD.Get_rank()
            except ImportError:
                MPI = None
                mpi_rank = 0
                logger.warn('Running with a single MPI process. This should work, but the results may differ from the ones publshed in Plappert et al.')

            if hasattr(env, '_max_episode_steps'):
                max_episode_steps = env._max_episode_steps
            else:
                max_episode_steps = default_max_episode_steps # otherwise use defaulit max episode steps
            env =  Monitor(env,
                           os.path.join(logger.get_dir(), str(mpi_rank) + '.' + str(subrank)),
                           allow_early_resets=True)
            # hack to re-expose _max_episode_steps (ideally should replace reliance on it downstream)
            env = gym.wrappers.TimeLimit(env, max_episode_steps=max_episode_steps)
        return env

    kwargs['make_env'] = make_env
    tmp_env = cached_make_env(kwargs['make_env'])
    max_episode_steps = default_max_episode_steps
    if hasattr(tmp_env, '_max_episode_steps'):
        max_episode_steps = tmp_env._max_episode_steps
    elif hasattr(tmp_env, 'env') and hasattr(tmp_env.env, '_max_episode_steps'):
        max_episode_steps = tmp_env.env._max_episode_steps

    kwargs['T'] = max_episode_steps

    kwargs['max_u'] = np.array(kwargs['max_u']) if isinstance(kwargs['max_u'], list) else kwargs['max_u']
    kwargs['gamma'] = 1. - 1. / kwargs['T']
    if 'lr' in kwargs:
        kwargs['pi_lr'] = kwargs['lr']
        kwargs['Q_lr'] = kwargs['lr']
        del kwargs['lr']
    for name in ['buffer_size', 'hidden', 'layers','network_class','polyak','batch_size', 
                 'Q_lr', 'pi_lr', 'norm_eps', 'norm_clip', 'max_u','action_l2', 'clip_obs', 
                 'scope', 'relative_goals', 'use_supervised', 'use_huber', 'delta', 'grad_clip_value']:
        wgcsl_params[name] = kwargs[name]
        kwargs['_' + name] = kwargs[name]
        del kwargs[name]
    
    kwargs['wgcsl_params'] = wgcsl_params
    return kwargs


def log_params(params, logger=logger):
    for key in sorted(params.keys()):
        logger.info('{}: {}'.format(key, params[key]))


def configure_her(params):
    env = cached_make_env(params['make_env'])
    env.reset()

    def reward_fun(ag_2, g, info):  # vectorized
        return env.compute_reward(achieved_goal=ag_2, desired_goal=g, info=info)

    # Prepare configuration for HER.
    her_params = {
        'reward_fun': reward_fun,
        'no_relabel': params['no_relabel']
    }
    for name in ['replay_strategy', 'replay_k']:
        her_params[name] = params[name]
        params['_' + name] = her_params[name]
        del params[name]

    sample_supervised, her_sampler = make_sample_transitions(**her_params)
    random_sampler = make_random_sample(her_params['reward_fun'])
    samplers = {
        'random': random_sampler,
        'her': her_sampler,
        'supervised':sample_supervised
    }
    return samplers, reward_fun

def simple_goal_subtract(a, b):
    assert a.shape == b.shape
    return a - b

def configure_wgcsl(dims, params, reuse=False, use_mpi=True, clip_return=True, offline_train=False):
    samplers, reward_fun = configure_her(params)
    # Extract relevant parameters.
    rollout_batch_size = params['rollout_batch_size']
    wgcsl_params = params['wgcsl_params']

    input_dims = dims.copy()
    # WGCSL agent
    env = cached_make_env(params['make_env'])
    env.reset()
    wgcsl_params.update({'input_dims': input_dims,  # agent takes an input observations
                        'T': params['T'],
                        'clip_pos_returns': True,  # clip positive returns
                        'clip_return': (1. / (1. - params['gamma'])) if clip_return else np.inf,  # max abs of return 
                        'rollout_batch_size': rollout_batch_size,
                        'subtract_goals': simple_goal_subtract,
                        'sample_transitions': samplers['her'],
                        'random_sampler':samplers['random'],
                        'supervised_sampler':samplers['supervised'],
                        'gamma': params['gamma'],
                        'su_method': params['su_method'],
                        'baw_delta': params['baw_delta'],
                        'baw_max': params['baw_max'],
                        })
    wgcsl_params['info'] = {
        'env_name': params['env_name'],
        'reward_fun':reward_fun
    } 
    policy = WGCSL(reuse=reuse, **wgcsl_params, use_mpi=use_mpi, offline_train=offline_train)  
    return policy


def configure_dims(params):
    env = cached_make_env(params['make_env'])
    env.reset()
    obs, _, _, info = env.step(env.action_space.sample())
    dims = {
        'o': obs['observation'].shape[0],
        'u': env.action_space.shape[0],
        'g': obs['desired_goal'].shape[0],
    }
    return dims
