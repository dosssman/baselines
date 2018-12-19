'''
Disclaimer: this code is highly based on trpo_mpi at @openai/baselines and @openai/imitation
'''

import argparse
import os
import os.path as osp
import logging
from mpi4py import MPI
from tqdm import tqdm

import numpy as np
import datetime, time
import gym
import tempfile

from baselines.gail import mlp_policy
from baselines.common import set_global_seeds, tf_util as U
from baselines.common.misc_util import boolean_flag
from baselines import bench
from baselines import logger
from baselines.gail.dataset.mujoco_dset import Mujoco_Dset
from baselines.gail.adversary import TransitionClassifier
import tensorflow as tf

from baselines.gail.gym_torcs import TorcsEnv

def argsparser():
    parser = argparse.ArgumentParser("Tensorflow Implementation of GAIL")
    parser.add_argument('--env_id', help='environment ID', default='Hopper-v2')
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--expert_path', type=str, default='data/deterministic.trpo.Hopper.0.00.npz')
    parser.add_argument('--checkpoint_dir', help='the directory to save model', default='checkpoint')
    parser.add_argument('--log_dir', help='the directory to save log file', default='log')
    parser.add_argument('--load_model_path', help='if provided, load the model', type=str, default=None)
    # Task
    parser.add_argument('--task', type=str, choices=['train', 'evaluate', 'sample'], default='train')
    # for evaluatation
    boolean_flag(parser, 'stochastic_policy', default=True, help='use stochastic/deterministic policy to evaluate')
    boolean_flag(parser, 'save_sample', default=False, help='save the trajectories or not')
    #  Mujoco Dataset Configuration
    parser.add_argument('--traj_limitation', type=int, default=-1)
    # Optimization Configuration
    parser.add_argument('--g_step', help='number of steps to train policy in each epoch', type=int, default=3)
    parser.add_argument('--d_step', help='number of steps to train discriminator in each epoch', type=int, default=1)
    # Network Configuration (Using MLP Policy)
    parser.add_argument('--policy_hidden_size', type=int, default=100)
    parser.add_argument('--adversary_hidden_size', type=int, default=100)
    # Algorithms Configuration
    parser.add_argument('--algo', type=str, choices=['trpo', 'ppo'], default='trpo')
    parser.add_argument('--max_kl', type=float, default=0.01)
    parser.add_argument('--policy_entcoeff', help='entropy coefficiency of policy', type=float, default=0)
    parser.add_argument('--adversary_entcoeff', help='entropy coefficiency of discriminator', type=float, default=1e-3)
    # Traing Configuration
    parser.add_argument('--save_per_iter', help='save model every xx iterations', type=int, default=100)
    parser.add_argument('--num_timesteps', help='number of timesteps per episode', type=int, default=5e6)
    # Behavior Cloning
    boolean_flag(parser, 'pretrained', default=False, help='Use BC to pretrain')
    parser.add_argument('--BC_max_iter', help='Max iteration for training BC', type=int, default=1e4)
    return parser.parse_args()


def get_task_name(args):
    task_name = args.algo + "_gail."
    if args.pretrained:
        task_name += "with_pretrained."
    if args.traj_limitation != np.inf:
        task_name += "transition_limitation_%d." % args.traj_limitation
    task_name += args.env_id.split("-")[0]
    task_name = task_name + ".g_step_" + str(args.g_step) + ".d_step_" + str(args.d_step) + \
        ".policy_entcoeff_" + str(args.policy_entcoeff) + ".adversary_entcoeff_" + str(args.adversary_entcoeff)
    task_name += ".seed_" + str(args.seed)
    return task_name


def main(args):
    U.make_session(num_cpu=1).__enter__()
    set_global_seeds(args.seed)

    # XXX: Hook up Gym Torcs
    vision = False
    throttle = True
    gear_change = False
    # race_config_path = os.path.dirname(os.path.abspath(__file__)) + \
    #     "/raceconfig/agent_practice.xml"

    # Agent10Fixed_Sparse


    # Agent10Fixed_Sparse A Speedway
    race_config_path = os.path.dirname(os.path.abspath(__file__)) + \
    "/raceconfig/agent_10fixed_sparsed_aspeedway_var_1.xml"

    # Agent 10 Fixed Second track First Variation
    race_config_path = os.path.dirname(os.path.abspath(__file__)) + \
    "/raceconfig/agent_10fixed_sparsed_eroad_var_1.xml"

    # Agent 10 Fixed Second track First Variation
    race_config_path = os.path.dirname(os.path.abspath(__file__)) + \
    "/raceconfig/agent_10fixed_sparsed_eroad_var_2.xml"

    # Agent 10 Fixed Second track First Variation
    race_config_path = os.path.dirname(os.path.abspath(__file__)) + \
    "/raceconfig/agent_10fixed_track1_var3.xml"

    # Agent10Fixed_Sparse
    race_config_path = os.path.dirname(os.path.abspath(__file__)) + \
    "/raceconfig/agent_10fixed_sparsed_1.xml"

    race_config_path = os.path.dirname(os.path.abspath(__file__)) + \
    "/raceconfig/agent_10fixed_sparsed_4.xml"

    # Agent 10 Fixed Second track First Variation
    race_config_path = os.path.dirname(os.path.abspath(__file__)) + \
    "/raceconfig/agent_10fixed_sparsed_forza_var_1.xml"

    # Agent 10 Fixed Second track First Variation
    race_config_path = os.path.dirname(os.path.abspath(__file__)) + \
    "/raceconfig/agent_10fixed_sparsed_forza_var_2.xml"

    # Agent 10 Fixed Second track First Variation
    race_config_path = os.path.dirname(os.path.abspath(__file__)) + \
    "/raceconfig/agent_10fixed_sparsed_forza_var_1.xml"

    # Agent 10 Fixed Second track First Variation
    race_config_path = os.path.dirname(os.path.abspath(__file__)) + \
    "/raceconfig/agent_10fixed_sparsed_track_2_var_2.xml" # Baboss

    # Agent 10 Fixed Second track First Variation
    race_config_path = os.path.dirname(os.path.abspath(__file__)) + \
    "/raceconfig/agent_10fixed_sparsed_track_2_var_1.xml"

    # Agent 10 Fixed Second track First Variation
    race_config_path = os.path.dirname(os.path.abspath(__file__)) + \
    "/raceconfig/agent_10fixed_sparsed_track_2_var_3.xml"

    rendering = True
    noisy = False

    # TODO: How Restrict to 3 laps when evaling ?
    lap_limiter = 2

    # env = gym.make(args.env_id)
    env = TorcsEnv(vision=vision, throttle=True, gear_change=False,
		race_config_path=race_config_path, rendering=rendering,
		lap_limiter = lap_limiter, noisy=noisy)

    # 2 hidden layers policy
    def policy_fn(name, ob_space, ac_space, reuse=False):
        return mlp_policy.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
            reuse=reuse, hid_size=args.policy_hidden_size, num_hid_layers=2)

    # 3 hid layers policy
    # def policy_fn(name, ob_space, ac_space, reuse=False):
    #     return mlp_policy.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
    #         reuse=reuse, hid_size=args.policy_hidden_size, num_hid_layers=3)

    # XXX: Intuitive log folder, probably save weihts there too & params overide
    task_name = get_task_name( args)

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
    # logger.configure( args.log_dir)
    # print( "# DEBUG: Logging to %s" % dir)

    # XXX Eval reparams
    args.stochastic_policy = False
    args.task = "evaluate"
    args.load_model_path = os.path.join( args.log_dir, "checkpoint")
    # args.load_model_path = os.path.join( args.load_model_path,
    #     "trpo_gail.transition_limitation_-1.Hopper.g_step_3.d_step_1.policy_entcoeff_0.adversary_entcoeff_0.001.seed_0" +
    #     "/trpo_gail.transition_limitation_-1.Hopper.g_step_3.d_step_1.policy_entcoeff_0.adversary_entcoeff_0.001.seed_0")

    # args.load_model_path = "/home/z3r0/random/rl/openai_logs/openai-gailtorcs/Doss10Fixed_110eps_GAILed/checkpoint/torcs_gail/torcs_gail"
    # args.load_model_path = "/home/z3r0/random/rl/openai_logs/openai-gailtorcs/Doss10Fixed_130eps_NoSlice_GAILed_MildlyStrict_TimestepPerBatchto1024_MaxKL_0.05/checkpoint/torcs_gail/torcs_gail_642"
    # args.load_model_path = "/home/z3r0/random/rl/openai_logs/openai-gailtorcs/DossCtrl10Fixed_100eps_GAILed_MaxRew3e4PLus/checkpoint/torcs_gail/torcs_gail_771"


    # Doss Ctrl 100 Episode most promising so far 2018-10-24
    # args.load_model_path = "/home/z3r0/random/rl/openai_logs/openai-gailtorcs/DossCtrl10Fixed_170eps_BC_GAILed/checkpoint/torcs_gail/torcs_gail_1040"

    # DossCtrl10Fixed_170eps_BC_GAILed_NoSlice
    # args.load_model_path = "/home/z3r0/random/rl/openai_logs/openai-gailtorcs/DossCtrl10Fixed_Series/DossCtrl10Fixed_170eps_BC_GAILed_NoSlice/checkpoint/torcs_gail/torcs_gail_145"
    # args.load_model_path = "/home/z3r0/random/rl/openai_logs/openai-gailtorcs/DossCtrl10Fixed_Series/DossCtrl10Fixed_170eps_BC_GAILed_NoSlice/checkpoint/torcs_gail/torcs_gail_193"
    # args.load_model_path = "/home/z3r0/random/rl/openai_logs/openai-gailtorcs/DossCtrl10Fixed_Series/DossCtrl10Fixed_170eps_BC_GAILed_NoSlice/checkpoint/torcs_gail/torcs_gail_286"
    # args.load_model_path = "/home/z3r0/random/rl/openai_logs/openai-gailtorcs/DossCtrl10Fixed_Series/DossCtrl10Fixed_170eps_BC_GAILed_NoSlice/checkpoint/torcs_gail/torcs_gail_350"
    # args.load_model_path = "/home/z3r0/random/rl/openai_logs/openai-gailtorcs/DossCtrl10Fixed_Series/DossCtrl10Fixed_170eps_BC_GAILed_NoSlice/checkpoint/torcs_gail/torcs_gail_530"
    # args.load_model_path = "/home/z3r0/random/rl/openai_logs/openai-gailtorcs/DossCtrl10Fixed_Series/DossCtrl10Fixed_170eps_BC_GAILed_NoSlice/checkpoint/torcs_gail/torcs_gail_783"
    # args.load_model_path = "/home/z3r0/random/rl/openai_logs/openai-gailtorcs/DossCtrl10Fixed_Series/DossCtrl10Fixed_170eps_BC_GAILed_NoSlice/checkpoint/torcs_gail/torcs_gail_900"
    # args.load_model_path = "/home/z3r0/random/rl/openai_logs/openai-gailtorcs/DossCtrl10Fixed_Series/DossCtrl10Fixed_170eps_BC_GAILed_NoSlice/checkpoint/torcs_gail/torcs_gail_1050"

    # Round 2 with NoSlice an traj sample at 3600
    # args.load_model_path = "/home/z3r0/random/rl/openai_logs/openai-gailtorcs/DossCtrl10Fixed_Series/DossCtrl10Fixed_170eps_BC_GAILed_NoSlice_Contd/checkpoint/torcs_gail/torcs_gail_292" # Actually ok on second track
    # args.load_model_path = "/home/z3r0/random/rl/openai_logs/openai-gailtorcs/DossCtrl10Fixed_Series/DossCtrl10Fixed_170eps_BC_GAILed_NoSlice_Contd/checkpoint/torcs_gail/torcs_gail_345" # Actually ok on second track

    # Roundd 3 with NoSlice and traj sample at 3600, bootstrape from Rouns 1
    # args.load_model_path = "/home/z3r0/random/rl/openai_logs/openai-gailtorcs/DossCtrl10Fixed_170eps_BC_GAILed_NoSlice_Contd2/checkpoint/torcs_gail/torcs_gail_107"
    # args.load_model_path = "/home/z3r0/random/rl/openai_logs/openai-gailtorcs/DossCtrl10Fixed_170eps_BC_GAILed_NoSlice_Contd2/checkpoint/torcs_gail/torcs_gail_205" # Ok on track 2 var 1
    # args.load_model_path = "/home/z3r0/random/rl/openai_logs/openai-gailtorcs/DossCtrl10Fixed_170eps_BC_GAILed_NoSlice_Contd2/checkpoint/torcs_gail/torcs_gail_220"
    # args.load_model_path = "/home/z3r0/random/rl/openai_logs/openai-gailtorcs/DossCtrl10Fixed_170eps_BC_GAILed_NoSlice_Contd2/checkpoint/torcs_gail/torcs_gail_329" # Ok on track 2 var 1
    # args.load_model_path = "/home/z3r0/random/rl/openai_logs/openai-gailtorcs/DossCtrl10Fixed_170eps_BC_GAILed_NoSlice_Contd2/checkpoint/torcs_gail/torcs_gail_346"

    # Doss 10 Fixed Analogs
    # 1 eps + DETERMINSITRC POLICY MF
    # args.load_model_path = "/home/z3r0/random/rl/openai_logs/openai-gailtorcs/Doss10FixedAnal_1ep/checkpoint/torcs_gail/torcs_gail_500"
    # args.load_model_path = "/home/z3r0/random/rl/openai_logs/openai-gailtorcs/Doss10FixedAnal_1ep/checkpoint/torcs_gail/torcs_gail_900"
    # args.load_model_path = "/home/z3r0/random/rl/openai_logs/openai-gailtorcs/Doss10FixedAnal_1ep/checkpoint/torcs_gail/torcs_gail_1050"

    # # 70 eps + STOCH POLICY
    # args.load_model_path = "/home/z3r0/random/rl/openai_logs/openai-gailtorcs/Doss10FixedAnal_70eps/checkpoint/torcs_gail/torcs_gail_900"
    # args.load_model_path = "/home/z3r0/random/rl/openai_logs/openai-gailtorcs/Doss10FixedAnal_70eps/checkpoint/torcs_gail/torcs_gail_1110"
    #
    # # 130 eps + STOCH POLICY
    # args.load_model_path = "/home/z3r0/random/rl/openai_logs/openai-gailtorcs/Doss10FixedAnal_130eps/checkpoint/torcs_gail/torcs_gail_500"
    # args.load_model_path = "/home/z3r0/random/rl/openai_logs/openai-gailtorcs/Doss10FixedAnal_130eps/checkpoint/torcs_gail/torcs_gail_900"
    # args.load_model_path = "/home/z3r0/random/rl/openai_logs/openai-gailtorcs/Doss10FixedAnal_130eps/checkpoint/torcs_gail/torcs_gail_1160"
    #
    # # 200 eps
    # args.load_model_path = "/home/z3r0/random/rl/openai_logs/openai-gailtorcs/Doss10FixedAnal_200eps/checkpoint/torcs_gail/torcs_gail_750"
    # args.load_model_path = "/home/z3r0/random/rl/openai_logs/openai-gailtorcs/Doss10FixedAnal_200eps/checkpoint/torcs_gail/torcs_gail_900"
    # args.load_model_path = "/home/z3r0/random/rl/openai_logs/openai-gailtorcs/Doss10FixedAnal_200eps/checkpoint/torcs_gail/torcs_gail_936"
    # args.load_model_path = "/home/z3r0/random/rl/openai_logs/openai-gailtorcs/Doss10FixedAnal_200eps/checkpoint/torcs_gail/torcs_gail_1100"
    #
    # # 200 eps
    # args.load_model_path = "/home/z3r0/random/rl/openai_logs/openai-gailtorcs/Doss10FixedAnal_200eps_3layers_2/checkpoint/torcs_gail/torcs_gail_1200"
    # args.load_model_path = "/home/z3r0/random/rl/openai_logs/openai-gailtorcs/Doss10FixedAnal_200eps_3layers_2/checkpoint/torcs_gail/torcs_gail_1360"
    # args.load_model_path = "/home/z3r0/random/rl/openai_logs/openai-gailtorcs/Doss10FixedAnal_200eps_3layers_2/checkpoint/torcs_gail/torcs_gail_1800"
    # args.load_model_path = "/home/z3r0/random/rl/openai_logs/openai-gailtorcs/Doss10FixedAnal_200eps_3layers_2/checkpoint/torcs_gail/torcs_gail_2335"


    # Testing REMI
    # args.load_model_path = "/home/z3r0/random/rl/openai_logs/openai-remi/DossCtrl_DDPGCkpt560_NoSlice_alpha_4_Run6/checkpoint/torcs_remi/torcs_remi_410" # Slow + Vibration
    # args.load_model_path = "/home/z3r0/random/rl/openai_logs/openai-remi/DossCtrl_DDPGCkpt560_NoSlice_alpha_4_Run5/checkpoint/torcs_remi/torcs_remi_250" # Slow but effective
    # args.load_model_path = "/home/z3r0/random/rl/openai_logs/openai-remi/DossCtrl_DDPGCkpt560_NoSlice_alpha_4_Run5/checkpoint/torcs_remi/torcs_remi_131" # Passes the 3rd corner + Almost goes to the end

    # 200eps over 5m timnesteps First effective run
    # args.load_model_path = "/home/z3r0/random/rl/openai_logs/openai-gailtorcs/Doss10FixedAnal_200eps_5mTsteps/checkpoint/torcs_gail/torcs_gail_1200"
    # args.load_model_path = "/home/z3r0/random/rl/openai_logs/openai-gailtorcs/Doss10FixedAnal_200eps_5mTsteps/checkpoint/torcs_gail/torcs_gail_1500" # Boy this guy good, can go to the second corner
    # args.load_model_path = "/home/z3r0/random/rl/openai_logs/openai-gailtorcs/Doss10FixedAnal_200eps_5mTsteps/checkpoint/torcs_gail/torcs_gail_1700" # Boy this guy good, can go to the second corner but crash
    # args.load_model_path = "/home/z3r0/random/rl/openai_logs/openai-gailtorcs/Doss10FixedAnal_200eps_5mTsteps/checkpoint/torcs_gail/torcs_gail_1900" # Boy this guy good, can go to the second corner but crash
    # args.load_model_path = "/home/z3r0/random/rl/openai_logs/openai-gailtorcs/Doss10FixedAnal_200eps_5mTsteps/checkpoint/torcs_gail/torcs_gail_2000" # Boy this guy good, can go to the second corner but crash
    # args.load_model_path = "/home/z3r0/random/rl/openai_logs/openai-gailtorcs/Doss10FixedAnal_200eps_5mTsteps/checkpoint/torcs_gail/torcs_gail_2700" # Boy this guy good, can go to the second corner but crash
    # args.load_model_path = "/home/z3r0/random/rl/openai_logs/openai-gailtorcs/Doss10FixedAnal_200eps_5mTsteps/checkpoint/torcs_gail/torcs_gail_2700" # Boy this guy good, can go to the second corner but crash
    # args.load_model_path = "/home/z3r0/random/rl/openai_logs/openai-gailtorcs/Doss10FixedAnal_200eps_5mTsteps/checkpoint/torcs_gail/torcs_gail_2732" # Boy this guy good, can go to the second corner but crash

    # 200eps Doss10FixedAnal_200eps_5mTsteps_Contd1
    # args.load_model_path = "/home/z3r0/random/rl/openai_logs/openai-gailtorcs/Doss10FixedAnal_200eps_5mTsteps_Contd1/checkpoint/torcs_gail/torcs_gail_100"
    # args.load_model_path = "/home/z3r0/random/rl/openai_logs/openai-gailtorcs/Doss10FixedAnal_200eps_5mTsteps_Contd1/checkpoint/torcs_gail/torcs_gail_600"
    # args.load_model_path = "/home/z3r0/random/rl/openai_logs/openai-gailtorcs/Doss10FixedAnal_200eps_5mTsteps_Contd1/checkpoint/torcs_gail/torcs_gail_675"
    # args.load_model_path = "/home/z3r0/random/rl/openai_logs/openai-gailtorcs/Doss10FixedAnal_200eps_5mTsteps_Contd1/checkpoint/torcs_gail/torcs_gail_806"
    # args.load_model_path = "/home/z3r0/random/rl/openai_logs/openai-gailtorcs/Doss10FixedAnal_200eps_5mTsteps_Contd1/checkpoint/torcs_gail/torcs_gail_825"

    # 200eps Doss10FixedAnal + DDPG Agent Mixed model
    # args.load_model_path = "/home/z3r0/random/rl/openai_logs/openai-remi/Doss10FixedAnal_DDPG_Chkp560_200eps/checkpoint/torcs_remi/torcs_remi_1200"

    # 200eps Doss10FixedAnal_200eps_5mTsteps_Contd2
    # args.load_model_path = "/home/z3r0/random/rl/openai_logs/openai-gailtorcs/Doss10FixedAnal_200eps_5mTsteps_Contd2/checkpoint/torcs_gail/torcs_gail_500"
    # args.load_model_path = "/home/z3r0/random/rl/openai_logs/openai-remi/Doss10FixedAnal_DDPG_Chkp560_200eps_Run2/checkpoint/torcs_remi/torcs_remi_483"

    # 200eps Doss10FixedAnal_200eps_Run3
    # args.load_model_path = "/home/z3r0/random/rl/openai_logs/openai-gailtorcs/Doss10FixedAnal_200eps_Run3/checkpoint/torcs_gail/torcs_gail_3100"

    # Remi Run2
    # args.load_model_path = "/home/z3r0/random/rl/openai_logs/openai-remi/Doss10FixedAnal_DDPG_Chkp560_200eps_Run2/checkpoint/torcs_remi/torcs_remi_1000"
    # args.load_model_path = "/home/z3r0/random/rl/openai_logs/openai-remi/Doss10FixedAnal_DDPG_Chkp560_200eps_Run2/checkpoint/torcs_remi/torcs_remi_1610"
    # args.load_model_path = "/home/z3r0/random/rl/openai_logs/openai-remi/Doss10FixedAnal_DDPG_Chkp560_200eps_Run2/checkpoint/torcs_remi/torcs_remi_2084"

    # GailTorcs Run5
    # args.load_model_path = "/home/z3r0/random/rl/openai_logs/openai-gailtorcs/Doss10FixedAnal_200eps_Run5/checkpoint/torcs_gail/torcs_gail_791"
    # args.load_model_path = "/home/z3r0/random/rl/openai_logs/openai-gailtorcs/Doss10FixedAnal_200eps_Run5/checkpoint/torcs_gail/torcs_gail_859"
    # args.load_model_path = "/home/z3r0/random/rl/openai_logs/openai-gailtorcs/Doss10FixedAnal_200eps_Run5/checkpoint/torcs_gail/torcs_gail_929"

    # GailTorcs Run5 Contd
    # args.load_model_path = "/home/z3r0/random/rl/openai_logs/openai-gailtorcs/Doss10FixedAnal_200eps_Run5_Contd/checkpoint/torcs_gail/torcs_gail_985"
    # args.load_model_path = "/home/z3r0/random/rl/openai_logs/openai-gailtorcs/Doss10FixedAnal_200eps_Run5_Contd/checkpoint/torcs_gail/torcs_gail_1646"
    # args.load_model_path = "/home/z3r0/random/rl/openai_logs/openai-gailtorcs/Doss10FixedAnal_200eps_Run5_Contd/checkpoint/torcs_gail/torcs_gail_1649"
    # args.load_model_path = "/home/z3r0/random/rl/openai_logs/openai-gailtorcs/Doss10FixedAnal_200eps_Run5_Contd/checkpoint/torcs_gail/torcs_gail_2090"
    # args.load_model_path = "/home/z3r0/random/rl/openai_logs/openai-gailtorcs/Doss10FixedAnal_200eps_Run5_Contd/checkpoint/torcs_gail/torcs_gail_2255"
    # args.load_model_path = "/home/z3r0/random/rl/openai_logs/openai-gailtorcs/Doss10FixedAnal_200eps_Run5_Contd/checkpoint/torcs_gail/torcs_gail_3375"
    # args.load_model_path = "/home/z3r0/random/rl/openai_logs/openai-gailtorcs/Doss10FixedAnal_200eps_Run5_Contd/checkpoint/torcs_gail/torcs_gail_4978"
    # args.load_model_path = "/home/z3r0/random/rl/openai_logs/openai-gailtorcs/Doss10FixedAnal_200eps_Run5_Contd/checkpoint/torcs_gail/torcs_gail_7538"
    # args.load_model_path = "/home/z3r0/random/rl/openai_logs/openai-gailtorcs/Doss10FixedAnal_200eps_Run5_Contd/checkpoint/torcs_gail/torcs_gail_6501"

    # Gail Run6
    # args.load_model_path = "/home/z3r0/random/rl/openai_logs/openai-gailtorcs/Doss10FixedAnal_200eps_Run6/checkpoint/torcs_gail/torcs_gail_1641"
    # args.load_model_path = "/home/z3r0/random/rl/openai_logs/openai-gailtorcs/Doss10FixedAnal_200eps_Run6/checkpoint/torcs_gail/torcs_gail_2006"
    # args.load_model_path = "/home/z3r0/random/rl/openai_logs/openai-gailtorcs/Doss10FixedAnal_200eps_Run6/checkpoint/torcs_gail/torcs_gail_3911"
    # args.load_model_path = "/home/z3r0/random/rl/openai_logs/openai-gailtorcs/Doss10FixedAnal_200eps_Run6/checkpoint/torcs_gail/torcs_gail_5661"
    # args.load_model_path = "/home/z3r0/random/rl/openai_logs/openai-gailtorcs/Doss10FixedAnal_200eps_Run6/checkpoint/torcs_gail/torcs_gail_7729"

    # Remi Run3
    # args.load_model_path = "/home/z3r0/random/rl/openai_logs/openai-remi/Doss10FixedAnal_DDPG_Chkp560_200eps_Run3/checkpoint/torcs_remi/torcs_remi_1868" # Full Track
    # args.load_model_path = "/home/z3r0/random/rl/openai_logs/openai-remi/Doss10FixedAnal_DDPG_Chkp560_200eps_Run3/checkpoint/torcs_remi/torcs_remi_1972" # Full Track
    # args.load_model_path = "/home/z3r0/random/rl/openai_logs/openai-remi/Doss10FixedAnal_DDPG_Chkp560_200eps_Run3/checkpoint/torcs_remi/torcs_remi_2014" # Full track, drifter

    # Remi Run4
    args.load_model_path = "/home/z3r0/random/rl/openai_logs/openai-remi/Doss10FixedAnal_DDPG_Chkp560_200eps_Run4/checkpoint/torcs_remi/torcs_remi_2336"
    # args.load_model_path = "/home/z3r0/random/rl/openai_logs/openai-remi/Doss10FixedAnal_DDPG_Chkp560_200eps_Run4/checkpoint/torcs_remi/torcs_remi_2451"
    # args.load_model_path = "/home/z3r0/random/rl/openai_logs/openai-remi/Doss10FixedAnal_DDPG_Chkp560_200eps_Run4/checkpoint/torcs_remi/torcs_remi_5381"
    # args.load_model_path = "/home/z3r0/random/rl/openai_logs/openai-remi/Doss10FixedAnal_DDPG_Chkp560_200eps_Run4/checkpoint/torcs_remi/torcs_remi_5785"
    # args.load_model_path = "/home/z3r0/random/rl/openai_logs/openai-remi/Doss10FixedAnal_DDPG_Chkp560_200eps_Run4/checkpoint/torcs_remi/torcs_remi_6158"

    print( "# DEBUG: Model path: ", args.load_model_path)
    args.stochastic_policy = True

    # env = bench.Monitor(env, logger.get_dir() and
    #                     osp.join(logger.get_dir(), "monitor.csv"), allow_early_resets=True)
    env.seed(args.seed)
    # gym.logger.setLevel(logging.WARN)
    # task_name = get_task_name(args)
    # args.checkpoint_dir = osp.join(args.checkpoint_dir, task_name)
    # args.log_dir = osp.join(args.log_dir, task_name)

    # XXX Default params override
    args.expert_path = os.path.join( args.log_dir,
        "data/DossCtrl10Fixed_170eps/expert_data.npz")
    # task_name = get_task_name( args)
    # args.checkpoint_dir = os.path.join( args.log_dir, "checkpoint")
    # args.checkpoint_dir = os.path.join( args.checkpoint_dir, task_name)
    # assert isinstance(args.checkpoint_dir, str)
    # os.makedirs(args.checkpoint_dir, exist_ok=True)
    # Training time ( hopefully) and timestep constraints
    # Save samples
    args.save_sample = False

    if args.task == 'train':
        dataset = Mujoco_Dset(expert_path=args.expert_path, traj_limitation=args.traj_limitation)
        reward_giver = TransitionClassifier(env, args.adversary_hidden_size, entcoeff=args.adversary_entcoeff)
        train(env,
              args.seed,
              policy_fn,
              reward_giver,
              dataset,
              args.algo,
              args.g_step,
              args.d_step,
              args.policy_entcoeff,
              args.num_timesteps,
              args.save_per_iter,
              args.checkpoint_dir,
              args.log_dir,
              args.pretrained,
              args.BC_max_iter,
              task_name
              )
    elif args.task == 'evaluate':
        runner(env,
               policy_fn,
               args.load_model_path,
               timesteps_per_batch=1024,
               number_trajs=10,
               stochastic_policy=args.stochastic_policy,
               save=args.save_sample
               )
    else:
        raise NotImplementedError
    env.close()


def train(env, seed, policy_fn, reward_giver, dataset, algo,
          g_step, d_step, policy_entcoeff, num_timesteps, save_per_iter,
          checkpoint_dir, log_dir, pretrained, BC_max_iter, task_name=None):

    pretrained_weight = None
    if pretrained and (BC_max_iter > 0):
        # Pretrain with behavior cloning
        from baselines.gail import behavior_clone
        pretrained_weight = behavior_clone.learn(env, policy_fn, dataset,
                                                 max_iters=BC_max_iter)

    if algo == 'trpo':
        from baselines.gail import trpo_mpi
        # Set up for MPI seed
        rank = MPI.COMM_WORLD.Get_rank()
        # if rank != 0:
        #     logger.set_level(logger.DISABLED)
        workerseed = seed + 10000 * MPI.COMM_WORLD.Get_rank()
        set_global_seeds(workerseed)
        env.seed(workerseed)
        trpo_mpi.learn(env, policy_fn, reward_giver, dataset, rank,
                       pretrained=pretrained, pretrained_weight=pretrained_weight,
                       g_step=g_step, d_step=d_step,
                       entcoeff=policy_entcoeff,
                       max_timesteps=num_timesteps,
                       ckpt_dir=checkpoint_dir, log_dir=log_dir,
                       save_per_iter=save_per_iter,
                       timesteps_per_batch=1024,
                       max_kl=0.01, cg_iters=10, cg_damping=0.1,
                       gamma=0.995, lam=0.97,
                       vf_iters=5, vf_stepsize=1e-3,
                       task_name=task_name)
    else:
        raise NotImplementedError


def runner(env, policy_func, load_model_path, timesteps_per_batch, number_trajs,
           stochastic_policy, save=False, reuse=False):

    # Setup network
    # ----------------------------------------
    ob_space = env.observation_space
    ac_space = env.action_space
    pi = policy_func("pi", ob_space, ac_space, reuse=reuse)
    U.initialize()
    # Prepare for rollouts
    # ----------------------------------------
    U.load_state(load_model_path)

    print( "Model Path: %s" % load_model_path)

    # U.load_variables(load_model_path, variables=pi.get_variables(),
    #     sess=tf.get_default_session())

    obs_list = []
    acs_list = []
    len_list = []
    ret_list = []
    for _ in tqdm(range(number_trajs)):
        traj = traj_1_generator(pi, env, timesteps_per_batch, stochastic=stochastic_policy)
        obs, acs, ep_len, ep_ret = traj['ob'], traj['ac'], traj['ep_len'], traj['ep_ret']
        obs_list.append(obs)
        acs_list.append(acs)
        len_list.append(ep_len)
        ret_list.append(ep_ret)
    if stochastic_policy:
        print('stochastic policy:')
    else:
        print('deterministic policy:')
    if save:
        # Don't need env.spec.id, also not declared in GymTorcsEnv specs
        filename = load_model_path.split('/')[-1] + '.' # + env.spec.id
        np.savez(filename, obs=np.array(obs_list), acs=np.array(acs_list),
                 lens=np.array(len_list), rets=np.array(ret_list))
    avg_len = sum(len_list)/len(len_list)
    avg_ret = sum(ret_list)/len(ret_list)
    print("Average length:", avg_len)
    print("Average return:", avg_ret)
    return avg_len, avg_ret


# Sample one trajectory (until trajectory end)
def traj_1_generator(pi, env, horizon, stochastic):

    t = 0
    ac = env.action_space.sample()  # not used, just so we have the datatype
    new = True  # marks if we're on first timestep of an episode

    ob = env.reset()
    cur_ep_ret = 0  # return in current episode
    cur_ep_len = 0  # len of current episode

    # Initialize history arrays
    obs = []
    rews = []
    news = []
    acs = []
    start_time = time.time()

    while True:
        ac, vpred = pi.act(stochastic, ob)
        # print( "Timestep %d - Steer: %.3f - Accel: %.3f" % (t, ac[0], ac[1]))
        obs.append(ob)
        news.append(new)
        acs.append(ac)

        ob, rew, new, _ = env.step(ac)
        rews.append(rew)

        cur_ep_ret += rew
        cur_ep_len += 1
        if new or t >= horizon:
            break
        t += 1

    duration = time.time() - start_time

    print( "### DEBUG: Duration %f" % (duration))
    print( "### DEBUG: Episode length %d" % cur_ep_len)
    print( "### DEBUG: Episode Score %d" % cur_ep_ret)
    print( "### DEBUG: Sampling rate %f" % ( cur_ep_len / duration))

    obs = np.array(obs)
    rews = np.array(rews)
    news = np.array(news)
    acs = np.array(acs)
    traj = {"ob": obs, "rew": rews, "new": news, "ac": acs,
            "ep_ret": cur_ep_ret, "ep_len": cur_ep_len}
    return traj


if __name__ == '__main__':
    args = argsparser()
    main(args)
