from collections import OrderedDict

import numpy as np
import tensorflow as tf
from collections import defaultdict
from tensorflow.contrib.staging import StagingArea

from wgcsl.common import logger
from wgcsl.algo.util import (
    import_function, store_args, flatten_grads, transitions_in_episode_batch)
from wgcsl.algo.normalizer import Normalizer
from wgcsl.algo.replay_buffer import ReplayBuffer
from wgcsl.common.mpi_adam import MpiAdam
from wgcsl.common import tf_util


def dims_to_shapes(input_dims):
    return {key: tuple([val]) if val > 0 else tuple() for key, val in input_dims.items()}

def huber_loss(target, v_to_goal, delta=1.0):
    error = target - v_to_goal
    abs_error = tf.abs(error)

    # Calculate Huber loss
    quadratic = tf.minimum(abs_error, delta)
    linear = abs_error - quadratic
    loss = 0.5 * quadratic**2 + delta * linear

    return loss

def expectile_loss(target, v_to_goal, tau):
    diff = target - v_to_goal
    condition = tf.cast((diff > 0), dtype=tf.float32)
    return (condition * tau + (1 - condition) * (1 - tau)) * tf.square(diff)

def expectile_huber_loss(target, v_to_goal, tau, delta=1.0):
    diff = target - v_to_goal
    condition = tf.cast((diff > 0), dtype=tf.float32)
    abs_diff = tf.abs(diff)
    squared_loss = 0.5 * tf.square(abs_diff)
    linear_loss = delta * (abs_diff - 0.5 * delta)
    
    loss = tf.where(abs_diff < delta, squared_loss, linear_loss)
    return (condition * tau + (1 - condition) * (1 - tau)) * loss

class WGCSL(object):
    @store_args
    def __init__(self, input_dims, buffer_size, hidden, layers, network_class, polyak, batch_size,
                 Q_lr, pi_lr, norm_eps, norm_clip, max_u, action_l2, clip_obs, scope, T,
                 rollout_batch_size, subtract_goals, relative_goals, clip_pos_returns, clip_return,
                 sample_transitions, random_sampler, gamma,  supervised_sampler, use_supervised, su_method,
                reuse=False, offline_train=False, use_huber=False, delta=1.0, grad_clip_value=-1, **kwargs):
        """Implementation of policy with value funcion that is used in combination with WGCSL
        """
        if self.clip_return is None:
            self.clip_return = np.inf

        self.create_actor_critic = import_function(self.network_class)
        input_shapes = dims_to_shapes(self.input_dims)
        self.dimo = self.input_dims['o']
        self.dimg = self.input_dims['g']
        self.dimu = self.input_dims['u']

        # Prepare staging area for feeding data to the model. 
        stage_shapes = OrderedDict()
        for key in sorted(self.input_dims.keys()):
            if key.startswith('info_'):
                continue
            stage_shapes[key] = (None, *input_shapes[key])
        for key in ['o', 'g']:
            stage_shapes[key + '_2'] = stage_shapes[key]
        stage_shapes['r'] = (None,)
        self.stage_shapes = stage_shapes

        # Create network.
        with tf.variable_scope(self.scope):
            self.staging_tf = StagingArea(
                dtypes=[tf.float32 for _ in self.stage_shapes.keys()],
                shapes=list(self.stage_shapes.values()))
            self.buffer_ph_tf = [
                tf.placeholder(tf.float32, shape=shape) for shape in self.stage_shapes.values()]
            self.stage_op = self.staging_tf.put(self.buffer_ph_tf)
            self._create_network(reuse=reuse)

        # Configure the replay buffer.
        buffer_shapes = {key: (self.T-1 if key != 'o' else self.T, *input_shapes[key]) for key, val in input_shapes.items()}
        buffer_shapes['g'] = (buffer_shapes['g'][0], self.dimg)
        buffer_shapes['ag'] = (self.T, self.dimg)
        # buffer_size % rollout_batch_size should be zero
        buffer_size = (self.buffer_size // self.rollout_batch_size) * self.rollout_batch_size 

        if self.use_supervised:
            sampler = self.supervised_sampler
            info = {
                'use_supervised':True,
                'gamma':self.gamma,
                'train_policy':self.train_policy,
                'get_Q_pi':self.get_Q_pi,
                'get_Q': self.get_Q,
                'method': self.su_method,
                'baw_delta':self.baw_delta,
                'baw_max': self.baw_max,
            }
        else: # for HER
            sampler = self.sample_transitions
            info = {}
        
        self.buffer = ReplayBuffer(buffer_shapes, buffer_size, self.T, sampler, self.sample_transitions, info)

        self.loss_stats = defaultdict(list)
        self.grad_stats = defaultdict(list)
        self.debug_stats = defaultdict(list)

    def _random_action(self, n):
        return np.random.uniform(low=-self.max_u, high=self.max_u, size=(n, self.dimu))

    def _preprocess_og(self, o, ag, g, ):
        if self.relative_goals:
            g_shape = g.shape
            g = g.reshape(-1, self.dimg)
            ag = ag.reshape(-1, self.dimg)
            g = self.subtract_goals(g, ag)
            g = g.reshape(*g_shape)
        o = np.clip(o, -self.clip_obs, self.clip_obs)
        g = np.clip(g, -self.clip_obs, self.clip_obs)
        return o, g

    def step(self, obs):
        actions = self.get_actions(obs['observation'], obs['achieved_goal'], obs['desired_goal'], use_target_net=use_target_net)
        return actions, None, None, None

    def action_only(self, o, g):
        o = np.clip(o, -self.clip_obs, self.clip_obs)
        g = np.clip(g, -self.clip_obs, self.clip_obs)

        policy = self.target  #self.target if use_target_net else
        action = self.sess.run(policy.pi_tf, feed_dict={
            policy.o_tf: o.reshape(-1, self.dimo),
            policy.g_tf: g.reshape(-1, self.dimg)
        })
        return action

    def get_actions(self, o, ag, g, noise_eps=0., random_eps=0., use_target_net=False,
                    compute_Q=False):
        o, g = self._preprocess_og(o, ag, g)
        policy = self.target if use_target_net else self.main
        if self.use_supervised:
            policy = self.main
        # values to compute
        vals = [policy.pi_tf]
        if compute_Q:
            vals += [policy.Q_pi_tf, policy.IQR_pi_tf]
        # feed
        feed = {
            policy.o_tf: o.reshape(-1, self.dimo),
            policy.g_tf: g.reshape(-1, self.dimg),
            policy.u_tf: np.zeros((o.size // self.dimo, self.dimu), dtype=np.float32)
        }

        ret = self.sess.run(vals, feed_dict=feed)
        # action postprocessing
        u = ret[0]
        noise = noise_eps * self.max_u * np.random.randn(*u.shape)  # gaussian noise
        u += noise
        u = np.clip(u, -self.max_u, self.max_u)
        u += np.random.binomial(1, random_eps, u.shape[0]).reshape(-1, 1) * (self._random_action(u.shape[0]) - u)  # eps-greedy
        if u.shape[0] == 1:
            u = u[0]
        u = u.copy()
        ret[0] = u

        if len(ret) == 1:
            return ret[0]
        else:
            return ret
    
    def get_Q(self, o, g, u):
        o = np.clip(o, -self.clip_obs, self.clip_obs)
        g = np.clip(g, -self.clip_obs, self.clip_obs)

        policy = self.main
        feed = {
            policy.o_tf: o.reshape(-1, self.dimo),
            policy.g_tf: g.reshape(-1, self.dimg),
            policy.u_tf: u.reshape(-1, self.dimu)
        }
        ret = self.sess.run(policy.Q_tf, feed_dict=feed)
        return ret

    def get_Q_pi(self, o, g):
        o = np.clip(o, -self.clip_obs, self.clip_obs)
        g = np.clip(g, -self.clip_obs, self.clip_obs)
        policy = self.main #self.target
        feed = {
            policy.o_tf: o.reshape(-1, self.dimo),
            policy.g_tf:g.reshape(-1, self.dimg)
        }
        ret = self.sess.run(policy.Q_pi_tf, feed_dict=feed)
        return ret

    def get_target_Q(self, o, g, a, ag):
        o, g = self._preprocess_og(o, ag, g)
        policy = self.main
        # feed
        feed = {
            policy.o_tf: o.reshape(-1, self.dimo),
            policy.g_tf: g.reshape(-1, self.dimg),
            policy.u_tf: np.zeros((o.size // self.dimo, self.dimu), dtype=np.float32) #??
        }

        ret = self.sess.run(policy.Q_tf, feed_dict=feed)
        return ret
    
    def get_loss_stat(self):
        logs = [('stats_loss/mean_' + k, np.mean(v)) for k,v in self.loss_stats.items()]
        self.loss_stats = defaultdict(list)
        return logs

    def get_grad_norm(self):
        logs = [('grad_norm/mean_' + k, np.mean(v)) for k,v in self.grad_stats.items()]
        self.grad_stats = defaultdict(list)
        return logs

    def get_debug_info(self):
        logs = [('debug/mean_' + k, np.mean(v)) for k,v in self.debug_stats.items()]
        self.debug_stats = defaultdict(list)
        return logs

    def store_episode(self, episode_batch, update_stats=True): #init=False
        """
        episode_batch: array of batch_size x (T or T+1) x dim_key 'o' is of size T+1, others are of size T
        """
        self.buffer.store_episode(episode_batch)
        if update_stats:
            # episode doesn't has key o_2
            episode_batch['o_2'] = episode_batch['o'][:, 1:, :]
            episode_batch['ag_2'] = episode_batch['ag'][:, 1:, :]
            num_normalizing_transitions = transitions_in_episode_batch(episode_batch)
            # add transitions to normalizer
            transitions = self.sample_transitions(episode_batch, num_normalizing_transitions)

            o, g, ag = transitions['o'], transitions['g'], transitions['ag']
            transitions['o'], transitions['g'] = self._preprocess_og(o, ag, g)
            # training normalizer online 
            self.o_stats.update(transitions['o'])
            self.g_stats.update(transitions['g'])
            self.o_stats.recompute_stats()
            self.g_stats.recompute_stats()
            self.u_stats.update(transitions['u'])
            self.u_stats.recompute_stats()

    def train_policy(self, o, g, u, weights=None):
        if weights is None:
            weights = np.ones(o.shape[0])
        
        pi_sl_loss, pi_sl_grad, pi_sl_grad_norm = self.sess.run(
            [self.policy_sl_loss, self.pi_sl_grad_tf, self.pi_sl_grad_norm],
            feed_dict={
                self.gcsl_weight_tf: weights,
                self.main.o_tf: o,
                self.main.g_tf: g,
                self.main.u_tf : u
            }
        )
        self.grad_stats["pi_grad_norm"].append(pi_sl_grad_norm)
        self.loss_stats["actor_loss"].append(pi_sl_loss)
        self.pi_adam.update(pi_sl_grad, self.pi_lr)
        return pi_sl_loss

    def _sync_optimizers(self):
        self.IQR_adam.sync()
        self.Q_adam.sync()
        self.pi_adam.sync()

    def _grads(self):
        # Avoid feed_dict here for performance!
        critic_loss, iqr_loss, actor_loss, Q_grad, IQR_grad, pi_grad,\
             Q_grad_norm, IQR_grad_norm, pi_grad_norm, iqr, abs_iqr, target_iqr, target_abs_iqr = self.sess.run([
            self.Q_loss_tf,
            self.IQR_loss_tf,
            self.pi_loss_tf,
            self.Q_grad_tf,
            self.IQR_grad_tf,
            self.pi_grad_tf,
            self.Q_grad_norm,
            self.IQR_grad_norm,
            self.pi_grad_norm,
            self.iqr,
            self.abs_iqr,
            self.target_iqr,
            self.target_abs_iqr
        ])

        self.loss_stats["actor_loss"].append(actor_loss)
        self.loss_stats["critic_loss"].append(critic_loss)
        self.grad_stats["q_grad_norm"].append(Q_grad_norm)
        self.grad_stats["iqr_grad_norm"].append(IQR_grad_norm)
        self.grad_stats["pi_grad_norm"].append(pi_grad_norm)

        self.debug_stats["iqr"].append(iqr)
        self.debug_stats["abs_iqr"].append(abs_iqr)
        self.debug_stats["target_iqr"].append(target_iqr)
        self.debug_stats["target_abs_iqr"].append(target_abs_iqr)

        return critic_loss, actor_loss, Q_grad, IQR_grad, pi_grad

    def _update(self, Q_grad, IQR_grad, pi_grad):
        self.Q_adam.update(Q_grad, self.Q_lr)
        self.IQR_adam.update(IQR_grad, self.Q_lr)
        self.pi_adam.update(pi_grad, self.pi_lr)


    def sample_batch(self, method='list'):
        transitions = self.buffer.sample(self.batch_size)  
        o, o_2, g = transitions['o'], transitions['o_2'], transitions['g']
        ag, ag_2 = transitions['ag'], transitions['ag_2']
        transitions['o'], transitions['g'] = self._preprocess_og(o, ag, g)
        transitions['o_2'], transitions['g_2'] = self._preprocess_og(o_2, ag_2, g)

        if self.offline_train:
            self.o_stats.update(transitions['o'])
            self.g_stats.update(transitions['g'])
            self.o_stats.recompute_stats()
            self.g_stats.recompute_stats()

        if method == 'list':
            transitions_batch = [transitions[key] for key in self.stage_shapes.keys()]
        else:
            transitions_batch = transitions
        return transitions_batch

    def stage_batch(self, batch=None):
        if batch is None:
            batch = self.sample_batch()
            self.temp_batch = batch
        assert len(self.buffer_ph_tf) == len(batch)
        self.sess.run(self.stage_op, feed_dict=dict(zip(self.buffer_ph_tf, batch)))

    def train(self, stage=True):
        if stage:
            self.stage_batch()
        if not self.use_supervised:
            critic_loss, actor_loss, Q_grad, IQR_grad, pi_grad = self._grads()
            self._update(Q_grad, IQR_grad, pi_grad)
            return critic_loss, actor_loss
        # WGCSL needs to learn the value function
        elif self.use_supervised and self.su_method not in ['', 'gamma']:
            self.update_critic_only()
        else:  # GCSL does not need to learn value function
            pass
    
    def update_critic_only(self):
        critic_loss, iqr_loss, Q_grad, IQR_grad,\
             Q_grad_norm, IQR_grad_norm, iqr, abs_iqr, target_iqr, target_abs_iqr = self.sess.run([
            self.Q_loss_tf,
            self.IQR_loss_tf,
            self.Q_grad_tf,
            self.IQR_grad_tf,
            self.Q_grad_norm,
            self.IQR_grad_norm,
            self.iqr,
            self.abs_iqr,
            self.target_iqr,
            self.target_abs_iqr
        ])
        self.loss_stats["critic_loss"].append(critic_loss)
        self.grad_stats["q_grad_norm"].append(Q_grad_norm)
        self.grad_stats["iqr_grad_norm"].append(IQR_grad_norm)

        self.debug_stats["iqr"].append(iqr)
        self.debug_stats["abs_iqr"].append(abs_iqr)
        self.debug_stats["target_iqr"].append(target_iqr)
        self.debug_stats["target_abs_iqr"].append(target_abs_iqr)

        self.Q_adam.update(Q_grad, self.Q_lr)
        self.IQR_adam.update(IQR_grad, self.Q_lr)

    def _init_target_net(self):
        self.sess.run(self.init_target_net_op)

    def update_target_net(self):
        self.sess.run(self.update_target_net_op)

    def clear_buffer(self):
        self.buffer.clear_buffer()

    def _vars(self, scope):
        res = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.scope + '/' + scope)
        assert len(res) > 0
        return res

    def _global_vars(self, scope):
        res = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.scope + '/' + scope)
        return res

    def _create_network(self, reuse=False):
        logger.info("Creating a WGCSL agent with action space %d x %s..." % (self.dimu, self.max_u))
        self.sess = tf_util.get_session()

        # running averages
        with tf.variable_scope('o_stats') as vs:
            if reuse:
                vs.reuse_variables()
            self.o_stats = Normalizer(self.dimo, self.norm_eps, self.norm_clip, sess=self.sess)
        with tf.variable_scope('g_stats') as vs:
            if reuse:
                vs.reuse_variables()
            self.g_stats = Normalizer(self.dimg, self.norm_eps, self.norm_clip, sess=self.sess)
        with tf.variable_scope('u_stats') as vs:
            if reuse:
                vs.reuse_variables()
            self.u_stats = Normalizer(self.dimu, self.norm_eps, self.norm_clip, sess=self.sess)

        # mini-batch sampling.
        batch = self.staging_tf.get()
        batch_tf = OrderedDict([(key, batch[i]) for i, key in enumerate(self.stage_shapes.keys())])
        batch_tf['r'] = tf.reshape(batch_tf['r'], [-1, 1])

        # networks
        with tf.variable_scope('main') as vs:
            if reuse:
                vs.reuse_variables()
            self.main = self.create_actor_critic(batch_tf, net_type='main', **self.__dict__)
            vs.reuse_variables()
        with tf.variable_scope('target') as vs:
            if reuse:
                vs.reuse_variables()
            target_batch_tf = batch_tf.copy()
            target_batch_tf['o'] = batch_tf['o_2']
            target_batch_tf['g'] = batch_tf['g_2']
            self.target = self.create_actor_critic(target_batch_tf, net_type='target', **self.__dict__)
            vs.reuse_variables()
        assert len(self._vars("main")) == len(self._vars("target"))

        # loss functions
        q_pi_to_goal = self.main.Q_pi_tf
        pi_to_goal = self.main.pi_tf
        q_to_goal = self.main.Q_tf
        iqr_q_to_goal = self.main.IQR_tf
        target_iqr_q_goal = self.target.IQR_tf

        self.iqr = tf.reduce_mean(tf.math.abs(iqr_q_to_goal[:, 0] - iqr_q_to_goal[:, 1]))
        self.abs_iqr = tf.reduce_mean(iqr_q_to_goal[:, 0] - iqr_q_to_goal[:, 1])
        self.target_iqr = tf.reduce_mean(tf.math.abs(target_iqr_q_goal[:, 0] - target_iqr_q_goal[:, 1]))
        self.target_abs_iqr = tf.reduce_mean(target_iqr_q_goal[:, 0] - target_iqr_q_goal[:, 1])

        target_Q_pi_tf = self.target.Q_pi_tf
        self.batch_r = batch_tf['r']
        clip_range = (-self.clip_return, self.clip_return)
        target_tf = tf.stop_gradient(tf.clip_by_value(batch_tf['r'] + self.gamma * target_Q_pi_tf, *clip_range))
        self.target_tf = target_tf
        if self.use_huber:
            # NOTE (lisheng) The quadratic loss has timed 1/2.
            per_Q_loss_tf = huber_loss(target_tf, q_to_goal, self.delta)

            iqr_loss_h1_tf = expectile_huber_loss(target_tf, iqr_q_to_goal[:, 0, None], 0.75, self.delta)
            iqr_loss_h2_tf = expectile_huber_loss(target_tf, iqr_q_to_goal[:, 1, None], 0.25, self.delta)
        else:
            per_Q_loss_tf = 0.5 * tf.square(target_tf - q_to_goal)

            iqr_loss_h1_tf = expectile_loss(target_tf, iqr_q_to_goal[:, 0, None], 0.75)
            iqr_loss_h2_tf = expectile_loss(target_tf, iqr_q_to_goal[:, 1, None], 0.25)

        self.Q_loss_tf = tf.reduce_mean(per_Q_loss_tf)
        self.IQR_loss_tf = tf.reduce_mean((iqr_loss_h1_tf + iqr_loss_h2_tf)/2.)

        self.pi_loss_tf = -tf.reduce_mean(q_pi_to_goal)
        self.pi_loss_tf += self.action_l2 * tf.reduce_mean(tf.square(pi_to_goal/ self.max_u))

        # training policy with supervised learning (GCSL)
        self.gcsl_weight_tf = tf.placeholder(tf.float32, shape=(None,) , name='weights')
        self.weighted_sl_loss = tf.reduce_mean(tf.square(self.main.u_tf - self.main.pi_tf),axis=1)
        self.policy_sl_loss = tf.reduce_mean(self.gcsl_weight_tf * self.weighted_sl_loss)  #  + 0.01 * self.temp_action_loss

        Q_grads_tf = tf.gradients(self.Q_loss_tf, self._vars('main/Q'))
        IQR_grads_tf = tf.gradients(self.IQR_loss_tf, self._vars('main/IQR'))
        pi_grads_tf = tf.gradients(self.pi_loss_tf, self._vars('main/pi'))
        pi_sl_grads_tf = tf.gradients(self.policy_sl_loss, self._vars('main/pi'))

        self.Q_grad_norm = tf.math.reduce_mean([tf.norm(grad) for grad in Q_grads_tf])
        self.IQR_grad_norm = tf.math.reduce_mean([tf.norm(grad) for grad in IQR_grads_tf])
        self.pi_grad_norm = tf.math.reduce_mean([tf.norm(grad) for grad in pi_grads_tf])
        self.pi_sl_grad_norm = tf.math.reduce_mean([tf.norm(grad) for grad in pi_sl_grads_tf])

        if self.grad_clip_value > 0:
            grad_clip_value = self.grad_clip_value
            Q_grads_tf = [tf.clip_by_value(grad, -grad_clip_value, grad_clip_value) for grad in Q_grads_tf]
            IQR_grads_tf = [tf.clip_by_value(grad, -grad_clip_value, grad_clip_value) for grad in IQR_grads_tf]
            pi_grads_tf = [tf.clip_by_value(grad, -grad_clip_value, grad_clip_value) for grad in pi_grads_tf]
            pi_sl_grads_tf = [tf.clip_by_value(grad, -grad_clip_value, grad_clip_value) for grad in pi_sl_grads_tf]

        assert len(self._vars('main/Q')) == len(Q_grads_tf)
        assert len(self._vars('main/IQR')) == len(IQR_grads_tf)
        assert len(self._vars('main/pi')) == len(pi_grads_tf)
        self.Q_grads_vars_tf = zip(Q_grads_tf, self._vars('main/Q'))
        self.IQR_grads_vars_tf = zip(IQR_grads_tf, self._vars('main/IQR'))
        self.pi_grads_vars_tf = zip(pi_grads_tf, self._vars('main/pi'))
        self.pi_sl_grads_vars_tf = zip(pi_sl_grads_tf, self._vars('main/pi'))

        self.Q_grad_tf = flatten_grads(grads=Q_grads_tf, var_list=self._vars('main/Q'))
        self.IQR_grad_tf = flatten_grads(grads=IQR_grads_tf, var_list=self._vars('main/IQR'))
        self.pi_grad_tf = flatten_grads(grads=pi_grads_tf, var_list=self._vars('main/pi'))
        self.pi_sl_grad_tf = flatten_grads(grads=pi_sl_grads_tf, var_list=self._vars('main/pi'))

        # optimizers
        self.Q_adam = MpiAdam(self._vars('main/Q'), scale_grad_by_procs=False)
        self.IQR_adam = MpiAdam(self._vars('main/IQR'), scale_grad_by_procs=False)
        self.pi_adam = MpiAdam(self._vars('main/pi'), scale_grad_by_procs=False)

        # polyak averaging
        self.main_vars = self._vars('main/Q') + self._vars('main/IQR') + self._vars('main/pi')
        self.target_vars = self._vars('target/Q') + self._vars('target/IQR') + self._vars('target/pi')
        self.stats_vars = self._global_vars('o_stats') + self._global_vars('g_stats')

        self.init_target_net_op = list(
            map(lambda v: v[0].assign(v[1]), zip(self.target_vars, self.main_vars)))
        self.update_target_net_op = list(
            map(lambda v: v[0].assign(self.polyak * v[0] + (1. - self.polyak) * v[1]), zip(self.target_vars, self.main_vars)))

        # initialize all variables
        tf.variables_initializer(self._global_vars('')).run()
        self._sync_optimizers()
        self._init_target_net()

    def logs(self, prefix=''):
        logs = []
        logs += [('stats_o/mean', np.mean(self.sess.run([self.o_stats.mean])))]
        logs += [('stats_o/std', np.mean(self.sess.run([self.o_stats.std])))]
        logs += [('stats_g/mean', np.mean(self.sess.run([self.g_stats.mean])))]
        logs += [('stats_g/std', np.mean(self.sess.run([self.g_stats.std])))]
        logs += [('stats_u/mean', np.mean(self.sess.run([self.u_stats.mean])))]
        logs += [('stats_u/std', np.mean(self.sess.run([self.u_stats.std])))]
        if prefix != '' and not prefix.endswith('/'):
            return [(prefix + '/' + key, val) for key, val in logs]
        else:
            return logs

    def save(self, save_path):
        tf_util.save_variables(save_path)

