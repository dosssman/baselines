'''
The code is used to train BC imitator, or pretrained GAIL imitator
'''

import argparse
import tempfile
import os, datetime
import os.path as osp
import gym
import logging
from tqdm import tqdm

import tensorflow as tf

from baselines.gail_torqs import mlp_policy
from baselines import bench
from baselines import logger
from baselines.common import set_global_seeds, tf_util as U
from baselines.common.misc_util import boolean_flag
from baselines.common.mpi_adam import MpiAdam
from baselines.gail_torqs.main import runner
from baselines.gail_torqs.dataset.mujoco_dset import Mujoco_Dset

from baselines.gail_torqs.gym_torcs import TorcsEnv
from baselines.common.distributions import make_pdtype
from baselines.acktr.utils import dense as aktr_dense
import tensorflow.contrib as tc

def argsparser():
    parser = argparse.ArgumentParser("Tensorflow Implementation of Behavior Cloning")
    parser.add_argument('--env_id', help='environment ID', default='Torcs')
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--expert_path', type=str, default='data/deterministic.trpo.Hopper.0.00.npz')
    parser.add_argument('--checkpoint_dir', help='the directory to save model', default='checkpoint')
    parser.add_argument('--log_dir', help='the directory to save log file', default='log')
    #  Mujoco Dataset Configuration
    parser.add_argument('--traj_limitation', type=int, default=-1)
    # Network Configuration (Using MLP Policy)
    parser.add_argument('--policy_hidden_size', type=int, default=100)
    # for evaluatation
    boolean_flag(parser, 'stochastic_policy', default=False, help='use stochastic/deterministic policy to evaluate')
    boolean_flag(parser, 'save_sample', default=False, help='save the trajectories or not')
    parser.add_argument('--BC_max_iter', help='Max iteration for training BC', type=int, default=1e5)
    return parser.parse_args()

# XXX TRASH
class TorcsActor( object):
    def __init__( self, ob_space, ac_space, name="actor", layer_norm=True,
        reuse=False, gaussian_fixed_var=False):
        self.name = name
        with tf.variable_scope(name):
            if reuse:
                tf.get_variable_scope().reuse_variables()
        self.layer_norm = layer_norm
        self.nb_actions = ac_space.shape[0]
        self.scope = tf.get_variable_scope().name
        self.pdtype = pdtype = make_pdtype(ac_space)

        with( tf.variable_scope( self.name)) as scope:
            if reuse:
                scope.reuse_variables()

            # x = tf.placeholder( dtype=tf.float32, shape=(65,1), name="inputs")
            sequence_length = None
            ob = U.get_placeholder(name="ob", dtype=tf.float32, shape=[sequence_length] + list(ob_space.shape))

            last_out = tf.layers.dense( ob, 64, name="input")
            if self.layer_norm:
                last_out = tc.layers.layer_norm(last_out, center=True, scale=True)
            last_out = tf.nn.relu(last_out, name="h1")

            last_out = tf.layers.dense(last_out, 64)
            if self.layer_norm:
                last_out = tc.layers.layer_norm(last_out, center=True, scale=True)
            last_out = tf.nn.relu(last_out, name="h2")

            last_out = tf.layers.dense(last_out, self.nb_actions, kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3))
            last_out = tf.nn.tanh(last_out, name="output")

        # MLP Policy similarity hack
        if gaussian_fixed_var and isinstance(ac_space, gym.spaces.Box):
            mean = aktr_dense(last_out, pdtype.param_shape()[0]//2, "polfinal", U.normc_initializer(0.01))
            logstd = tf.get_variable(name="logstd", shape=[1, pdtype.param_shape()[0]//2], initializer=tf.zeros_initializer())
            pdparam = tf.concat([mean, mean * 0.0 + logstd], axis=1)
        else:
            pdparam = aktr_dense(last_out, pdtype.param_shape()[0], "polfinal", U.normc_initializer(0.01))

        self.pd = pdtype.pdfromflat(pdparam)

        # change for BC
        stochastic = U.get_placeholder(name="stochastic", dtype=tf.bool, shape=())
        ac = U.switch(stochastic, self.pd.sample(), self.pd.mode())
        self.ac = ac
        self._act = U.function([stochastic, ob], [ac, self.vpred])
        # return x

    def act(self, stochastic, ob):
        ac1, vpred1 = self._act(stochastic, ob[None])
        return ac1[0], vpred1[0]

    def get_variables(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope)

    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)

    def get_initial_state(self):
        return []

def learn(env, policy_func, dataset, optim_batch_size=128, max_iters=1e4,
          adam_epsilon=1e-5, optim_stepsize=3e-4,
          ckpt_dir=None, log_dir=None, task_name=None,
          verbose=False):

    val_per_iter = int(max_iters/10)
    ob_space = env.observation_space
    ac_space = env.action_space
    # pi = policy_func("pi", ob_space, ac_space)  # Construct network for new policy
    pi = TorcsActor( ob_space, ac_space)
    # placeholder
    ob = U.get_placeholder_cached(name="ob")
    ac = pi.pdtype.sample_placeholder([None])
    stochastic = U.get_placeholder_cached(name="stochastic")
    loss = tf.reduce_mean(tf.square(ac-pi.ac))
    var_list = pi.get_trainable_variables()
    adam = MpiAdam(var_list, epsilon=adam_epsilon)
    lossandgrad = U.function([ob, ac, stochastic], [loss]+[U.flatgrad(loss, var_list)])

    U.initialize()
    adam.sync()
    logger.log("Pretraining with Behavior Cloning...")
    for iter_so_far in tqdm(range(int(max_iters))):
        ob_expert, ac_expert = dataset.get_next_batch(optim_batch_size, 'train')
        train_loss, g = lossandgrad(ob_expert, ac_expert, True)
        adam.update(g, optim_stepsize)
        if verbose and iter_so_far % val_per_iter == 0:
            ob_expert, ac_expert = dataset.get_next_batch(-1, 'val')
            val_loss, _ = lossandgrad(ob_expert, ac_expert, True)
            logger.log("Training loss: {}, Validation loss: {}".format(train_loss, val_loss))

    if ckpt_dir is None:
        savedir_fname = tempfile.TemporaryDirectory().name
    else:
        savedir_fname = osp.join(ckpt_dir, task_name)
    # U.save_state(savedir_fname, var_list=pi.get_variables())
    # XXX Just save everything already
    U.save_state(savedir_fname)
    return savedir_fname


def get_task_name(args):
    task_name = 'BC'
    task_name += '.{}'.format(args.env_id.split("-")[0])
    task_name += '.traj_limitation_{}'.format(args.traj_limitation)
    task_name += ".seed_{}".format(args.seed)
    return task_name


def main(args):
    U.make_session(num_cpu=1).__enter__()
    set_global_seeds(args.seed)

    # env = gym.make(args.env_id)
    # XXX: Hook up Gym Torcs
    vision = False
    throttle = True
    gear_change = False
    race_config_path = os.path.dirname(os.path.abspath(__file__)) + \
        "/raceconfig/agent_practice.xml"
    rendering = False

    # TODO: How Restrict to 3 laps when evaling ?
    lap_limiter = 4

    # env = gym.make(args.env_id)
    env = TorcsEnv(vision=vision, throttle=True, gear_change=False,
		race_config_path=race_config_path, rendering=rendering,
		lap_limiter = lap_limiter)

    # XXX Log dir config
    dir = os.getenv('OPENAI_GEN_LOGDIR')
    if dir is None:
        dir = osp.join(tempfile.gettempdir(),
            datetime.datetime.now().strftime("openai-gailtorcs"))
    else:
        dir = osp.join( dir, datetime.datetime.now().strftime("openai-gailtorcs"))

    assert isinstance(dir, str)
    os.makedirs(dir, exist_ok=True)
    args.log_dir = dir
    # args.log_dir = osp.join(args.log_dir, task_name)
    # assert isinstance(args.log_dir, str)
    # os.makedirs(args.log_dir, exist_ok=True)
    logger.configure( dir=args.log_dir, format_strs="stdout,log,csv,tensorboard")
    print( "# DEBUG: Logging to %s" % logger.get_dir())

    def policy_fn(name, ob_space, ac_space, reuse=False):
        return mlp_policy.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
                                    reuse=reuse, hid_size=args.policy_hidden_size, num_hid_layers=2)
    env = bench.Monitor(env, logger.get_dir() and
                        osp.join(logger.get_dir(), "monitor.json"))
    env.seed(args.seed)
    gym.logger.setLevel(logging.WARN)
    task_name = get_task_name(args)
    # XXX Default params override
    args.expert_path = os.path.join( args.log_dir,
        "damned_200ep_1000step/expert_data.npz")
    task_name = get_task_name( args)
    args.checkpoint_dir = os.path.join( args.log_dir, "checkpoint")
    args.checkpoint_dir = os.path.join( args.checkpoint_dir, task_name)
    assert isinstance(args.checkpoint_dir, str)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # args.checkpoint_dir = osp.join(args.checkpoint_dir, task_name)
    # args.log_dir = osp.join(args.log_dir, task_name)
    args.stochastic_policy = False
    dataset = Mujoco_Dset(expert_path=args.expert_path, traj_limitation=args.traj_limitation)
    savedir_fname = learn(env,
                          policy_fn,
                          dataset,
                          max_iters=args.BC_max_iter,
                          ckpt_dir=args.checkpoint_dir,
                          log_dir=args.log_dir,
                          task_name=task_name,
                          verbose=True)
    avg_len, avg_ret = runner(env,
                              policy_fn,
                              savedir_fname,
                              timesteps_per_batch=1024,
                              number_trajs=10,
                              stochastic_policy=args.stochastic_policy,
                              save=args.save_sample,
                              reuse=True)


if __name__ == '__main__':
    args = argsparser()
    main(args)
