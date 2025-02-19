from collections import deque

import numpy as np
from functools import reduce

from wgcsl.algo.util import convert_episode_to_batch_major, store_args, discounted_return

GAMMA_ARRAY=None

def calcualte_discounted_returns(rewards, final_Q, gamma):
    returns = [final_Q]
    def accumulate(acc, reward):
        return [reward + gamma * acc[0]] + acc
    returns = reduce(accumulate, reversed(rewards.T), returns)[:-1]
    return np.array(returns).T

class RolloutWorker:

    @store_args
    def __init__(self, venv, policy, dims, logger, T, rollout_batch_size=1,
                 exploit=False, use_target_net=False, compute_Q=False, noise_eps=0,
                 random_eps=0, history_len=100, render=False, monitor=False, **kwargs):
        """Rollout worker generates experience by interacting with one or many environments.
        Args:
            venv: vectorized gym environments.
            policy (object): the policy that is used to act
            dims (dict of ints): the dimensions for observations (o), goals (g), and actions (u)
            logger (object): the logger that is used by the rollout worker
            rollout_batch_size (int): the number of parallel rollouts that should be used
            exploit (boolean): whether or not to exploit, i.e. to act optimally according to the
                current policy without any exploration
            use_target_net (boolean): whether or not to use the target net for rollouts
            compute_Q (boolean): whether or not to compute the Q values alongside the actions
            noise_eps (float): scale of the additive Gaussian noise
            random_eps (float): probability of selecting a completely random action
            history_len (int): length of history for statistics smoothing
            render (boolean): whether or not to render the rollouts
        """
        assert self.T > 0
        self.info_keys = [key.replace('info_', '') for key in dims.keys() if key.startswith('info_')]

        self.success_history = deque(maxlen=history_len)
        self.Q_history = deque(maxlen=history_len)
        self.return_history = deque(maxlen=history_len)
        self.dis_return_history = deque(maxlen=history_len)
        self.distances = deque(maxlen=history_len)

        self.n_episodes = 0
        self.reset_all_rollouts()
        self.clear_history()

    def reset_all_rollouts(self):
        self.obs_dict = self.venv.reset()
        self.initial_o = self.obs_dict['observation']
        self.initial_ag = self.obs_dict['achieved_goal']
        self.g = self.obs_dict['desired_goal']

    def generate_rollouts(self, random_ac=False, assign_goal=None):
        """Performs `rollout_batch_size` rollouts in parallel for time horizon `T` with the current
        policy acting on it accordingly.
        """
        self.reset_all_rollouts()

        # compute observations
        o = np.empty((self.rollout_batch_size, self.dims['o']), np.float32)  # observations
        ag = np.empty((self.rollout_batch_size, self.dims['g']), np.float32)  # achieved goals
        o[:] = self.initial_o
        ag[:] = self.initial_ag

        if assign_goal is not None:
            self.g = assign_goal

        # generate episodes
        obs, achieved_goals, acts, goals, successes, rewards = [], [], [], [], [], []
        dones = []
        info_values = [np.empty((self.T - 1, self.rollout_batch_size, self.dims['info_' + key]), np.float32) for key in self.info_keys]
        Qs = []
        IQRs = []
        for t in range(self.T):
            if random_ac:
                u = self.policy._random_action(self.rollout_batch_size)
            else:
                policy_output = self.policy.get_actions(
                    o, ag, self.g,
                    compute_Q=self.compute_Q,
                    noise_eps=self.noise_eps if not self.exploit else 0.,
                    random_eps=self.random_eps if not self.exploit else 0.,
                    use_target_net=self.use_target_net)

                if self.compute_Q:
                    u, Q, IQR_Q = policy_output
                    IQRs.append(IQR_Q[:, 0] - IQR_Q[:, 1])
                    Qs.append(Q)
                else:
                    u = policy_output

            if u.ndim == 1:
                # The non-batched case should still have a reasonable shape.
                u = u.reshape(1, -1)

            o_new = np.empty((self.rollout_batch_size, self.dims['o']))
            ag_new = np.empty((self.rollout_batch_size, self.dims['g']))
            success = np.zeros(self.rollout_batch_size)
            # compute new states and observations
            obs_dict_new, reward, done, info = self.venv.step(u)
            o_new = obs_dict_new['observation']
            ag_new = obs_dict_new['achieved_goal']
            success = np.array([i.get('is_success', 0.0) for i in info])

            if any(done) or t == self.T-1:
                # here we assume all environments are done is ~same number of steps, so we terminate rollouts whenever any of the envs returns done
                # trick with using vecenvs is not to add the obs from the environments that are "done", because those are already observations
                # after a reset
                break

            for i, info_dict in enumerate(info):
                for idx, key in enumerate(self.info_keys):
                    try:
                        info_values[idx][t, i] = info[i][key]
                    except:
                        pass

            if np.isnan(o_new).any():
                self.logger.warn('NaN caught during rollout generation. Trying again...')
                self.reset_all_rollouts()
                return self.generate_rollouts()

            dones.append(done)
            obs.append(o.copy())
            achieved_goals.append(ag.copy())
            successes.append(success.copy())
            acts.append(u.copy())
            goals.append(self.g.copy())
            rewards.append(reward.copy())
            o[...] = o_new
            ag[...] = ag_new
        obs.append(o.copy())
        achieved_goals.append(ag.copy())

        episode = dict(o=obs,
                       u=acts,
                       g=goals,
                       ag=achieved_goals,
                       r=rewards)

        for key, value in zip(self.info_keys, info_values):
            episode['info_{}'.format(key)] = value
        successful = np.array(successes)[-1, :]
        assert successful.shape == (self.rollout_batch_size,)
        success_rate = np.mean(successful)
        self.success_history.append(success_rate)
        dis_return, undis_return = discounted_return(rewards, self.gamma, reward_offset=False)
        self.return_history.append(undis_return)
        self.dis_return_history.append(dis_return)
        self.distances.append(np.linalg.norm(achieved_goals[-1] - goals[-1]))

        if self.compute_Q:
            self.Q_history.append(np.mean(Qs))
        self.n_episodes += self.rollout_batch_size
        # change shape to (rollout, steps, dim)

        if self.exploit:
            IQRA = np.stack(IQRs, axis=1)
            QA = np.concatenate(Qs, axis=1)
            SA = np.array(successful).T
            RA = np.array(rewards).T
            global GAMMA_ARRAY
            if not isinstance(GAMMA_ARRAY, np.ndarray):
                episode_length = RA.shape[1]
                GAMMA_ARRAY = np.power(self.gamma, np.arange(episode_length))
            TR = np.sum(RA*GAMMA_ARRAY, axis=1) + self.gamma * GAMMA_ARRAY[-1] * RA[:, -1] / (1 - self.gamma)
            discounted_final_Q = self.gamma * GAMMA_ARRAY[-1] * QA[:, -1]
            discounted_returns_plus_shifting_bias = calcualte_discounted_returns(RA, QA[:, -1], self.gamma)

            istb_bias = QA[:, 0] - TR
            shifting_bias = QA[:, -1] - RA[:, -1] / (1 - self.gamma)
            initial_shooting_bias = QA[:, 0] - TR - discounted_final_Q
            # the average one is incorrect.
            average_shooting_bias = np.mean(QA[:, :-1] - discounted_returns_plus_shifting_bias, axis=1)
            average_iqr = np.mean(IQRA, axis=1)
            return convert_episode_to_batch_major(episode), istb_bias, shifting_bias, initial_shooting_bias, \
                average_shooting_bias, average_iqr, IQRA, SA

        return convert_episode_to_batch_major(episode)  

    def clear_history(self):
        """Clears all histories that are used for statistics
        """
        self.success_history.clear()
        self.Q_history.clear()
        self.return_history.clear()
        self.dis_return_history.clear()

    def current_success_rate(self):
        return np.mean(self.success_history)

    def current_mean_Q(self):
        return np.mean(self.Q_history)
    
    def current_mean_return(self):
        return np.mean(self.return_history)

    def save_policy(self, path):
        """Pickles the current policy for later inspection.
        """
        self.policy.save(path)

    def logs(self, prefix='worker'):
        """Generates a dictionary that contains all collected statistics.
        """
        logs = []
        logs += [('success_rate', np.mean(self.success_history))]
        logs += [('return', np.mean(self.return_history))]
        logs += [('discount_return', np.mean(self.dis_return_history))]
        if self.compute_Q:
            logs += [('mean_Q', np.mean(self.Q_history))]
        logs += [('episode', self.n_episodes)]
        logs += [('distance', np.mean(self.distances))]
        logs += [('distance_std', np.std(self.distances))]

        if prefix != '' and not prefix.endswith('/'):
            return [(prefix + '/' + key, val) for key, val in logs]
        else:
            return logs