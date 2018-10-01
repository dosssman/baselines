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

from baselines.remi_torqs import main as run_torcs
from baselines.remi_torqs import mlp_policy
from baselines.common import set_global_seeds, tf_util as U
from baselines.common.misc_util import boolean_flag
from baselines.remi_torqs.dataset.mujoco_dset import Mujoco_Dset

from baselines.remi_torqs.gym_torcs import TorcsEnv

plt.style.use('ggplot')
CONFIG = {
    'traj_limitation': [1, 5, 10, 50],
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
    boolean_flag(parser, 'stochastic_policy', default=False, help='use stochastic/deterministic policy to evaluate')
    return parser.parse_args()


def evaluate_env(env_name, seed, policy_hidden_size, stochastic, reuse, prefix):

    def get_checkpoint_dir(checkpoint_list, limit, prefix):
        for checkpoint in checkpoint_list:
            if ('limitation_'+str(limit) in checkpoint) and (prefix in checkpoint):
                return checkpoint
        return None

    def policy_fn(name, ob_space, ac_space, reuse=False):
        return mlp_policy.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
                                    reuse=reuse, hid_size=policy_hidden_size, num_hid_layers=2)
    # XXX Readapt expert path
    dir = os.getenv('OPENAI_GEN_LOGDIR')
    # if dir is None:
    #     dir = osp.join(tempfile.gettempdir(),
    #         datetime.datetime.now().strftime("openai-gailtorcs"))
    # else:
    #     dir = osp.join( dir, datetime.datetime.now().strftime("openai-gailtorcs"))

    assert isinstance(dir, str)
    os.makedirs(dir, exist_ok=True)

    log_dir = dir
    expert_data = os.path.join( log_dir,
        "openai-remi/data/damned200ep720tstpInterpolated/expert_data.npz")
    rl_expert_data = os.path.join( log_dir,
        "openai-remi/data/ddpg200ep720tstpInterpolated/expert_data.npz")
    # data_path = os.path.join('data', 'deterministic.trpo.' + env_name + '.0.00.npz')
    dataset = load_dataset(expert_data)
    rl_dataset = load_dataset( rl_expert_data)
    # checkpoint_list = glob.glob(os.path.join('checkpoint', '*' + env_name + ".*"))
    log = {
        'traj_limitation': [],
        'upper_bound': [],
        "rl_upper_bound": [],
        'avg_ret': [],
        'avg_len': [],
        'normalized_ret': []
    }
    for i, limit in enumerate(CONFIG['traj_limitation']):
        # Do one evaluation
        upper_bound = sum(dataset.rets[:limit])/limit
        rl_upper_bound = sum(rl_dataset.rets[:limit])/limit
        # checkpoint_dir = get_checkpoint_dir(checkpoint_list, limit, prefix=prefix)
        # checkpoint_path = tf.train.latest_checkpoint(checkpoint_dir)
        # XXX Checkpoint path
        # 20180915~ ?
        # checkpoint_path = os.path.join( log_dir, "defiant/openai-remi/20180916_DamnedAndDDPGAlpha0_5GAILed/checkpoint/torcs_gail/torcs_gail_2300")

        # 20180917
        # checkpoint_path = os.path.join( log_dir, "openai-remi/mixedLossAlpha0_5_20180926/checkpoint/torcs_remi/torcs_remi_2450")

        # 20180917 Full RL through ReMi
        checkpoint_path = os.path.join( log_dir, "openai-remi/mixedLossAlpha0_0_20180928/checkpoint/torcs_remi/torcs_remi_2350")

        # 20180917 Full Expert through ReMi
        # checkpoint_path = os.path.join( log_dir, "openai-remi/mixedLossAlpha1_0_20180928/checkpoint/torcs_remi/torcs_remi_2350")

        # Alpha .3
        # checkpoint_path = os.path.join( log_dir, "openai-remi/mixedLossAlpha0_3_20180927/checkpoint/torcs_remi/torcs_remi_2350")

        # Alpha .7
        # checkpoint_path = os.path.join( log_dir, "openai-remi/mixedLossAlpha0_7_20180927/checkpoint/torcs_remi/torcs_remi_2350")

        print( "# DEBUG: Model path: ", (checkpoint_path + ".index"))
        # Not pretty but will do for now
        assert( os.path.isfile( checkpoint_path + ".index"))

        # env = gym.make(env_name + '-v1')
        # XXX: Custom env declaration
        vision = False
        throttle = True
        gear_change = False
        race_config_path = os.path.dirname(os.path.abspath(__file__)) + \
            "/raceconfig/agent_practice.xml"
        rendering = False
        lap_limiter = 4

        # env = gym.make(env_id)
        env = TorcsEnv(vision=vision, throttle=True, gear_change=False,
    		race_config_path=race_config_path, rendering=rendering,
    		lap_limiter = lap_limiter)

        env.seed(seed)
        print('Trajectory limitation: {}, Load checkpoint: {}, '.format(limit, checkpoint_path))
        # TODO: RUn Mujoco not meant to be used here
        avg_len, avg_ret = run_torcs.runner(env,
                                             policy_fn,
                                             checkpoint_path,
                                             timesteps_per_batch=1024,
                                             number_trajs=10,
                                             stochastic_policy=stochastic,
                                             reuse=((i != 0) or reuse))
        normalized_ret = avg_ret/upper_bound
        print('Upper bound: {}, RL Upper bound: {}, evaluation returns: {}, normalized scores: {}'.format(
            upper_bound, rl_upper_bound, avg_ret, normalized_ret))
        log['traj_limitation'].append(limit)
        log['upper_bound'].append(upper_bound)
        log["rl_upper_bound"].append( rl_upper_bound)
        log['avg_ret'].append(avg_ret)
        log['avg_len'].append(avg_len)
        log['normalized_ret'].append(normalized_ret)

        env.close()
    return log


def plot(env_name, bc_log, gail_log, stochastic):
    upper_bound = bc_log['upper_bound']
    rl_upper_bound = bc_log["rl_upper_bound"]
    # bc_avg_ret = bc_log['avg_ret']
    gail_avg_ret = gail_log['avg_ret']
    plt.plot(CONFIG['traj_limitation'], upper_bound)
    plt.plot(CONFIG['traj_limitation'], rl_upper_bound)
    # plt.plot(CONFIG['traj_limitation'], bc_avg_ret)
    plt.plot(CONFIG['traj_limitation'], gail_avg_ret)
    plt.xlabel('Number of expert trajectories')
    plt.ylabel('Accumulated reward')
    plt.title('{} unnormalized scores'.format(env_name))
    # plt.legend(['expert', "rl_expert", 'bc-imitator', 'gail-imitator'], loc='lower right')
    plt.legend(['expert', "rl_expert", 'remi-imitator'], loc='lower right')
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
    print('Evaluating {}'.format(args.env))
    bc_log = evaluate_env(args.env, args.seed, args.policy_hidden_size,
                          args.stochastic_policy, False, 'BC')
    print('Evaluation for {}'.format(args.env))
    print(bc_log)
    gail_log = evaluate_env(args.env, args.seed, args.policy_hidden_size,
                            args.stochastic_policy, True, 'gail')
    print('Evaluation for {}'.format(args.env))
    print(gail_log)
    plot(args.env, bc_log, gail_log, args.stochastic_policy)


if __name__ == '__main__':
    args = argsparser()
    main(args)
