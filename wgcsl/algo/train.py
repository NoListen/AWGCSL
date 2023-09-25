import os

import numpy as np
from mpi4py import MPI
import time
from copy import copy

from wgcsl.common import logger
from wgcsl.common import tf_util
from wgcsl.common.util import set_global_seeds
from wgcsl.common.mpi_moments import mpi_moments
import wgcsl.algo.config as config
from wgcsl.algo.rollout import RolloutWorker
from wgcsl.algo.util import dump_params

def mpi_average(value):
    if not isinstance(value, list):
        value = [value]
    if not any(value):
        value = [0.]
    return mpi_moments(np.array(value))[0]

def log_successful_trajectory_bias(bias_stat, success_bias_dict, success_array):
    if np.sum(success_array) != 0:
        for k, v in success_bias_dict.items():
            bias_stat["bias/success_" + k] = np.sum(v * success_array) / np.sum(success_array)
            bias_stat["bias/success_abs_" + k] = np.sum(np.abs(v) * success_array) / np.sum(success_array)
    else:
        for k, v in success_bias_dict.items():
            bias_stat["bias/success_" + k] = -1
            bias_stat["bias/success_abs_" + k] = -1

    return bias_stat

def log_failed_trajecotry_bias(bias_stat, failure_bias_dict, success_array):
    if np.min(success_array) == 0:
        for k, v in failure_bias_dict.items():
            bias_stat["bias/failure_" + k] = np.sum(v * (1 - success_array))/np.sum(1 - success_array)
            bias_stat["bias/failure_abs_" + k] = np.sum(np.abs(v) * (1 - success_array))/np.sum(1 - success_array)
    else:
        for k, v in failure_bias_dict.items():
            bias_stat["bias/failure_" + k] = -1
            bias_stat["bias/failure_abs_" + k] = -1
    return bias_stat
    

def evaluate(evaluator, logger, n_test_rollouts):
    evaluator.clear_history()
    evaluator.render = True
    bias_stat = {}
    istb_bias_list = []
    success_list = []
    shifting_bias_list = []
    initial_shooting_bias_list = []
    average_shooting_bias_list = []
    average_iqr_list = []
    per_step_iqr_array_list = []

    for _ in range(n_test_rollouts):
        _, istb_bias, shifting_bias, initial_shooting_bias, average_shooting_bias, \
            average_iqr, per_step_iqr_array, success = evaluator.generate_rollouts()
        istb_bias_list.append(istb_bias)
        shifting_bias_list.append(shifting_bias)
        initial_shooting_bias_list.append(initial_shooting_bias)
        average_shooting_bias_list.append(average_shooting_bias)
        success_list.append(success)
        average_iqr_list.append(average_iqr)
        per_step_iqr_array_list.append(per_step_iqr_array)

    istb_bias_array = np.concatenate(istb_bias_list)
    shifting_bias_array = np.concatenate(shifting_bias_list)
    initial_shooting_bias_array = np.concatenate(initial_shooting_bias_list)
    average_shooting_bias_array = np.concatenate(average_shooting_bias_list)
    success_array = np.concatenate(success_list)
    average_iqr_array = np.concatenate(average_iqr_list)
    all_per_step_iqr_array = np.concatenate(per_step_iqr_array_list)

    bias_stat["mean_abs_bias"] = np.mean(np.abs(istb_bias_array))
    bias_stat["average_bias"] = np.mean(istb_bias_array)
    bias_stat["average_iqr"] = np.mean(average_iqr_array)

    bias_data = {"initial": istb_bias_array, "shifting": shifting_bias_array, "initial_shooting": initial_shooting_bias_array,
                "average_shooting": average_shooting_bias_array, "average_iqr": average_iqr_array}

    bias_stat = log_successful_trajectory_bias(bias_stat, bias_data, success_array)
    bias_stat = log_failed_trajecotry_bias(bias_stat, bias_data, success_array)

    for key, val in bias_stat.items():
            logger.record_tabular(key, mpi_average(val))
    return all_per_step_iqr_array, success_array

def train(*, policy, rollout_worker, evaluator,
          n_epochs, n_test_rollouts, n_cycles, n_batches, policy_save_interval,
          save_path, random_init, play_no_training, offline_train, n_eps_per_cycle, **kwargs):
    rank = MPI.COMM_WORLD.Get_rank()

    if save_path and not play_no_training:
        latest_policy_path = os.path.join(save_path, 'policy_latest.pkl')
        best_policy_path = os.path.join(save_path, 'policy_best.pkl')
        periodic_policy_path = os.path.join(save_path, 'policy_{}.pkl')

    info_path = os.path.join(logger.get_dir(), "info.npz")

    # random_init for o/g/rnd stat and model training
    if random_init and not play_no_training and not offline_train:
        logger.info('Random initializing ...')
        rollout_worker.clear_history()
        for epi in range(int(random_init) // rollout_worker.rollout_batch_size): 
            episode = rollout_worker.generate_rollouts(random_ac=True)
            policy.store_episode(episode)

    best_success_rate = -1
    logger.info('Start training...')
    # num_timesteps = n_epochs * n_cycles * rollout_length * number of rollout workers
    
    n_rounds = n_eps_per_cycle//rollout_worker.rollout_batch_size
    remain = n_eps_per_cycle%rollout_worker.rollout_batch_size
    info_to_dump = {}
    for epoch in range(n_epochs):
        time_start = time.time()
        rollout_worker.clear_history()
        if remain != 0:
            print("WARNING: the actual episode for each batch is", n_rounds * rollout_worker.rollout_batch_size)
        for i in range(n_cycles):
            policy.dynamic_batch = False
            if remain != 0:
                print("WARNING: the actual episode for each batch is", n_rounds * rollout_worker.rollout_batch_size)
            if not offline_train:
                for _ in range(n_rounds):
                    episode = rollout_worker.generate_rollouts()
                    policy.store_episode(episode)
            for _ in range(n_batches):   
                policy.train()
            policy.update_target_net()

        # test
        evaluate(evaluator, logger, n_test_rollouts)

        all_per_step_iqr_array, success_array = evaluate(evaluator, logger, n_test_rollouts)
        info_to_dump["epoch_{}".format(epoch)] = {"iqr": all_per_step_iqr_array, "success": success_array}
        np.savez(info_path, **info_to_dump)

        for key, val in policy.get_loss_stat():
            logger.record_tabular(key, mpi_average(val))
        for key, val in policy.get_debug_info():
            logger.record_tabular(key, mpi_average(val))
        for key, val in policy.get_grad_norm():
            logger.record_tabular(key, mpi_average(val))

        for key, val in evaluator.logs('test'):
            logger.record_tabular(key, mpi_average(val))
        for key, val in rollout_worker.logs('train'):
            logger.record_tabular(key, mpi_average(val))
        for key, val in policy.logs():
            logger.record_tabular(key, mpi_average(val))

        # record logs
        time_end = time.time()
        logger.record_tabular('epoch', epoch)
        logger.record_tabular('epoch time(min)', (time_end - time_start)/60)

        if rank == 0:
            logger.dump_tabular()

        # save the policy if it's better than the previous ones
        success_rate = mpi_average(evaluator.current_success_rate())
        if rank == 0 and success_rate > best_success_rate and save_path and not play_no_training:
            best_success_rate = success_rate
            logger.info('New best success rate: {}. Saving policy to {} ...'.format(best_success_rate, best_policy_path))
            evaluator.save_policy(best_policy_path)
            evaluator.save_policy(latest_policy_path)
        if rank == 0 and policy_save_interval > 0 and epoch % policy_save_interval == 0 and save_path and not play_no_training:
            policy_path = periodic_policy_path.format(epoch)
            logger.info('Saving periodic policy to {} ...'.format(policy_path))
            evaluator.save_policy(policy_path)

        # make sure that different threads have different seeds
        local_uniform = np.random.uniform(size=(1,))
        root_uniform = local_uniform.copy()
        MPI.COMM_WORLD.Bcast(root_uniform, root=0)
        if rank != 0:
            assert local_uniform[0] != root_uniform[0]

    return policy


def learn(*, env, num_epoch, 
    seed=None,
    eval_env=None,
    replay_strategy='future',
    policy_save_interval=5,
    clip_return=True,
    demo_file=None,
    override_params=None,
    load_model=False,
    load_buffer=False,
    load_path=None,
    save_path=None,
    play_no_training=False,
    offline_train=False,
    mode=None,
    su_method='',
    **kwargs
):

    override_params = override_params or {} 
    rank = MPI.COMM_WORLD.Get_rank()
    if MPI is not None:
        rank = MPI.COMM_WORLD.Get_rank()
        num_cpu = MPI.COMM_WORLD.Get_size()

    # Seed everything.
    rank_seed = seed + 1000000 * rank if seed is not None else None
    set_global_seeds(rank_seed)

    # Prepare params.
    params = config.DEFAULT_PARAMS
    env_name = env.spec.id

    params['env_name'] = env_name
    params['replay_strategy'] = replay_strategy
    if env_name in config.DEFAULT_ENV_PARAMS:
        params.update(config.DEFAULT_ENV_PARAMS[env_name])  # merge env-specific parameters in
    params.update(**override_params)  # makes it possible to override any parameter

    params.update(kwargs)   # make kwargs part of params
    if 'num_epoch' in params:
        num_epoch = params['num_epoch']
    params['mode'] = mode
    params['su_method'] = su_method
    params = config.prepare_params(params)
    params['rollout_batch_size'] = env.num_envs
    random_init = params['random_init']
    # save total params
    dump_params(logger, params)

    if rank == 0:
        config.log_params(params, logger=logger)

    dims = config.configure_dims(params)
    policy = config.configure_wgcsl(action_space=copy(env.action_space), dims=dims, params=params, clip_return=clip_return, offline_train=offline_train)
    if load_path is not None:
        if load_model:
            tf_util.load_variables(os.path.join(load_path, 'policy_last.pkl'))
        if load_buffer:
            policy.buffer.load(os.path.join(load_path, 'buffer.pkl'))

    rollout_params = {
        'exploit': False,
        'use_target_net': False,
        'use_demo_states': True,
        'compute_Q': False,
        'T': params['T'],
    }
    eval_params = {
        'exploit': True,
        'use_target_net': params['test_with_polyak'],
        'use_demo_states': False,
        'compute_Q': True,
        'T': params['T'],
    }
    for name in ['T', 'rollout_batch_size', 'gamma', 'noise_eps', 'random_eps']:
        rollout_params[name] = params[name]
        eval_params[name] = params[name]

    eval_env = eval_env or env
    rollout_worker = RolloutWorker(env, policy, dims, logger, monitor=True, **rollout_params)
    evaluator = RolloutWorker(eval_env, policy, dims, logger, **eval_params)

    # no training
    if play_no_training:  
        # sample trajetories
        num_episode = 20 
        policy.buffer.clear_buffer()
        for _ in range(num_episode):
            episode = evaluator.generate_rollouts()
            policy.store_episode(episode)
        return policy

    return train(
        save_path=save_path, policy=policy, rollout_worker=rollout_worker,
        evaluator=evaluator, n_epochs=num_epoch, n_test_rollouts=params['n_test_rollouts'],
        n_cycles=params['n_cycles'], n_batches=params['n_batches'],
        policy_save_interval=policy_save_interval, demo_file=demo_file, random_init=random_init,
        play_no_training=play_no_training, offline_train=offline_train, n_eps_per_cycle=12)

