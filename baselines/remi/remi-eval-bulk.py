'''
This code is used to evalaute the imitators trained with different number of trajectories
and plot the results in the same figure for easy comparison.
'''

import argparse
import os
import os.path as osp
import glob
import gym
import datetime

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from baselines.remi import run_mujoco as run_torcs
from baselines.remi import mlp_policy
from baselines.common import set_global_seeds, tf_util as U
from baselines.common.misc_util import boolean_flag
from baselines.remi.dataset.mujoco_dset import Mujoco_Dset

from baselines.remi.gym_torcs import TorcsEnv
import pickle as pkl

plt.style.use('ggplot')
CONFIG = {
    # 'traj_limitation': [1, 5, 10, 50, 100, 200],
    'traj_limitation': [ 200],
}


def load_dataset(expert_path):
    dataset = Mujoco_Dset(expert_path=expert_path)
    return dataset


def argsparser():
    parser = argparse.ArgumentParser('Do evaluation')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--policy_hidden_size', type=int, default=100)
    parser.add_argument('--env', type=str, choices=['Hopper', 'Walker2d', 'HalfCheetah',
                                                    'Humanoid', 'HumanoidStandup'])
    boolean_flag(parser, 'stochastic_policy', default=True, help='use stochastic/deterministic policy to evaluate')
    return parser.parse_args()


def evaluate_env(env_name, seed, policy_hidden_size, stochastic, reuse, prefix, ckpt_path):

    def get_checkpoint_dir(checkpoint_list, limit, prefix):
        for checkpoint in checkpoint_list:
            if ('limitation_'+str(limit) in checkpoint) and (prefix in checkpoint):
                return checkpoint
        return None

    def policy_fn(name, ob_space, ac_space, reuse=tf.AUTO_REUSE):
        return mlp_policy.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
                                    reuse=reuse, hid_size=policy_hidden_size, num_hid_layers=2)
    # XXX Readapt expert path
    dir = os.getenv('OPENAI_GEN_LOGDIR')

    assert isinstance(dir, str)
    os.makedirs(dir, exist_ok=True)

    log_dir = dir

    data_path = os.path.join( log_dir,
        "openai-remi/data/Doss10FixedAnal_200eps_Sliced/expert_data.npz")
    # data_path = os.path.join('data', 'deterministic.trpo.' + env_name + '.0.00.npz')
    dataset = load_dataset(data_path)
    # checkpoint_list = glob.glob(os.path.join('checkpoint', '*' + env_name + ".*"))
    log = {
        'traj_limitation': [],
        'upper_bound': [],
        'avg_ret': [],
        'avg_len': [],
        'normalized_ret': [],
        'returns': []
    }
    for i, limit in enumerate(CONFIG['traj_limitation']):
        # Do one evaluation
        upper_bound = sum(dataset.rets[:limit])/limit
        args.load_model_path = ckpt_path

        # XXX: Custom env declaration
        vision = False
        throttle = True
        gear_change = False
        # Agent alone
        # race_config_path = os.path.dirname(os.path.abspath(__file__)) + \
        #     "/raceconfig/agent_practice.xml"

        # DamDamAgentFix
        race_config_path = os.path.dirname(os.path.abspath(__file__)) + \
            "/raceconfig/agent_10fixed_sparsed_4.xml"

        rendering = False
        lap_limiter = 2
        timestep_limit = 320

        # env = gym.make(env_id)
        env = TorcsEnv(vision=vision, throttle=True, gear_change=False,
    		race_config_path=race_config_path, rendering=rendering,
    		lap_limiter = lap_limiter)

        env.seed(seed)
        print('Trajectory limitation: {}, Load checkpoint: {}, '.format(limit,
            args.load_model_path))
        # TODO: RUn Mujoco not meant to be used here
        avg_len, avg_ret, max_ret, min_ret, log["returns"] = run_torcs.runner(env,
                                             policy_fn,
                                             args.load_model_path,
                                             timesteps_per_batch=3600,
                                             number_trajs=limit,
                                             stochastic_policy=stochastic,
                                             reuse=((i != 0) or reuse))

        normalized_ret = avg_ret/upper_bound
        print('Upper bound: {}, evaluation returns: {}, normalized scores: {}'.format(
            upper_bound, avg_ret, normalized_ret))
        log['traj_limitation'].append(limit)
        log['upper_bound'].append(upper_bound)
        log['avg_ret'].append(avg_ret)
        log['avg_len'].append(avg_len)
        log['max_ret'] = max_ret
        log['min_ret'] = min_ret
        # log['avg_avg_ret'] = np.mean( log['avg_ret'])
        # log['std_avg_ret'] = np.std( log['avg_ret'])
        log['normalized_ret'].append(normalized_ret)
        env.close()
    return log


def plot(env_name, bc_log, gail_log, stochastic):
    upper_bound = bc_log['upper_bound']
    bc_avg_ret = bc_log['avg_ret']
    gail_avg_ret = gail_log['avg_ret']
    plt.plot(CONFIG['traj_limitation'], upper_bound)
    plt.plot(CONFIG['traj_limitation'], bc_avg_ret)
    plt.plot(CONFIG['traj_limitation'], gail_avg_ret)
    plt.xlabel('Number of expert trajectories')
    plt.ylabel('Accumulated reward')
    plt.title('{} unnormalized scores'.format(env_name))
    plt.legend(['expert', 'bc-imitator', 'gail-imitator'], loc='lower right')
    plt.grid(b=True, which='major', color='gray', linestyle='--')
    if stochastic:
        title_name = '{}-unnormalized-stochastic-scores.png'.format(env_name)
    else:
        title_name = '{}-unnormalized-deterministic-scores.png'.format(env_name)

    # XXX Better logging policy ?
    dir = os.getenv('OPENAI_GEN_LOGDIR')
    if dir is None:
        dir = osp.join(tempfile.gettempdir(),
            datetime.datetime.now().strftime("openai-remi/result"))
    else:
        dir = osp.join( dir, datetime.datetime.now().strftime("openai-remi/result"))

    assert isinstance(dir, str)
    os.makedirs(dir, exist_ok=True)

    log_dir = dir
    title_name = osp.join( log_dir, title_name)
    plt.savefig(title_name)
    plt.close()

    bc_normalized_ret = bc_log['normalized_ret']
    gail_normalized_ret = gail_log['normalized_ret']
    plt.plot(CONFIG['traj_limitation'], np.ones(len(CONFIG['traj_limitation'])))
    plt.plot(CONFIG['traj_limitation'], bc_normalized_ret)
    plt.plot(CONFIG['traj_limitation'], gail_normalized_ret)
    plt.xlabel('Number of expert trajectories')
    plt.ylabel('Normalized performance')
    plt.title('{} normalized scores'.format(env_name))
    plt.legend(['expert', 'bc-imitator', 'gail-imitator'], loc='lower right')
    plt.grid(b=True, which='major', color='gray', linestyle='--')
    if stochastic:
        title_name = '{}-normalized-stochastic-scores.png'.format(env_name)
    else:
        title_name = '{}-normalized-deterministic-scores.png'.format(env_name)
    plt.ylim(0, 1.6)
    # XXX Better logging policy ?
    title_name = osp.join( log_dir, title_name)
    plt.savefig(title_name)
    plt.close()


def main(args):
    U.make_session(num_cpu=1).__enter__()
    set_global_seeds(args.seed)
    args.env = "Torcs ReMi"

    basedir = "/home/z3r0/random/rl/openai_logs/openai-remi"
    alphaplot_dir =  os.path.join( basedir, "alphaploting")

    if not os.path.exists( alphaplot_dir):
        os.makedirs( alphaplot_dir, exist_ok=True)

    ckpt_path = {
        # "0.0": "/home/z3r0/random/rl/openai_logs/openai-remi/Doss10FixedAnal_DDPG_Chkp560_200eps_Alpha_0.0_Run0/checkpoint/torcs_remi/torcs_remi_1367",
        # "0.1": "/home/z3r0/random/rl/openai_logs/openai-remi/Doss10FixedAnal_DDPG_Chkp560_200eps_Alpha_0.1_Run2/checkpoint/torcs_remi/torcs_remi_2828",
        # "0.2": "/home/z3r0/random/rl/openai_logs/openai-remi/Doss10FixedAnal_DDPG_Chkp560_200eps_Alpha_0.2_Run2/checkpoint/torcs_remi/torcs_remi_5463",
        # "0.3": "/home/z3r0/random/rl/openai_logs/openai-remi/Doss10FixedAnal_DDPG_Chkp560_200eps_Alpha_0.3_Run2/checkpoint/torcs_remi/torcs_remi_3248",
        # "0.4": "/home/z3r0/random/rl/openai_logs/openai-remi/Doss10FixedAnal_DDPG_Chkp560_200eps_Alpha_0.4_Run1/checkpoint/torcs_remi/torcs_remi_1256",
        # "0.5": "/home/z3r0/random/rl/openai_logs/openai-remi/Doss10FixedAnal_DDPG_Chkp560_200eps_Run3/checkpoint/torcs_remi/torcs_remi_2014",
        "0.6": "/home/z3r0/random/rl/openai_logs/openai-remi/Doss10FixedAnal_DDPG_Chkp560_200eps_Alpha_0.6_Run2/checkpoint/torcs_remi/torcs_remi_2461",
        "0.7": "/home/z3r0/random/rl/openai_logs/openai-remi/Doss10FixedAnal_DDPG_Chkp560_200eps_Alpha_0.7_Run3/checkpoint/torcs_remi/torcs_remi_1569",
        # "0.8": "/home/z3r0/random/rl/openai_logs/openai-remi/Doss10FixedAnal_DDPG_Chkp560_200eps_Alpha_0.8_Run2/checkpoint/torcs_remi/torcs_remi_3144",
        # "0.9": "/home/z3r0/random/rl/openai_logs/openai-remi/Doss10FixedAnal_DDPG_Chkp560_200eps_Alpha_0.9_Run2/checkpoint/torcs_remi/torcs_remi_3981",
        # "1.0": "/home/z3r0/random/rl/openai_logs/openai-remi/Doss10FixedAnal_DDPG_Chkp560_200eps_Alpha_1.0_Run0/checkpoint/torcs_remi/torcs_remi_4339"
    }

    eval_results = {
        # "0.0": {},
        # "0.1": {},
        # "0.2": {},
        # "0.3": {},
        # "0.4": {},
        # "0.5": {},
        # "0.6": {},
        # "0.7": {},
        # "0.8": {},
        # "0.9": {},
        # "1.0":{}
    }
    previousfilename = "alphastestres-2019-03-20-12-46-06-207014.pickle"
    previousfilename = os.path.join( alphaplot_dir, previousfilename)

    if os.path.exists( previousfilename):
        with open( previousfilename, "rb") as evalresfile:
            eval_results = pkl.load( evalresfile);

    set_global_seeds(args.seed)

    for ckpt_alpha, ckpt_path in ckpt_path.items():
        config = tf.ConfigProto(
            allow_soft_placement=True,
            inter_op_parallelism_threads=1,
            intra_op_parallelism_threads=1)
        config.gpu_options.allow_growth = True
        with tf.Session( config=config, graph=tf.Graph()):
            eval_results[ckpt_alpha] = evaluate_env(args.env, args.seed, args.policy_hidden_size,
                                    args.stochastic_policy, False, 'gail', ckpt_path = ckpt_path)
            # tf.reset_default_graph()

    # Save the Full log for later usage

    savefilename = os.path.join( alphaplot_dir,
        datetime.datetime.now().strftime("alphastestres-%Y-%m-%d-%H-%M-%S-%f.pickle"))

    with open( savefilename, "wb") as handle:
        pkl.dump( eval_results, handle, protocol=pkl.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    args = argsparser()
    main(args)
