import argparse
import time
import os
import logging
from baselines import logger, bench
from baselines.common.misc_util import (
    set_global_seeds,
    boolean_flag,
)

from baselines.ddpg_torqs.models import Actor, Critic
from baselines.ddpg_torqs.memory import Memory
from baselines.ddpg_torqs.noise import *

import gym
import tensorflow as tf
from mpi4py import MPI
import pickle

from baselines.ddpg.ddpg import DDPG
import baselines.common.tf_util as U
from collections import deque

# dosssman
from baselines.ddpg_torqs.gym_torcs import TorcsEnv
import csv # CHecking trace for GIAL

def run( seed, noise_type, layer_norm, nb_epochs, nb_epoch_cycles, reward_scale,
    normalize_returns, normalize_observations, critic_l2_reg, actor_lr, critic_lr,
    popart, gamma, clip_norm, nb_train_steps, nb_rollout_steps, nb_eval_steps, batch_size,
    tau=0.01, param_noise_adaption_interval=5, **kwargs):
    # Configure things.
    rank = MPI.COMM_WORLD.Get_rank()

    # XXX Do we still need the logger ? Kinda for the scores etc maybe ...
    # if rank != 0:
    #     logger.set_level(logger.DISABLED)

    # Create envs.
    # TODO: Hook up with gym torcs
    # Torcs Env Params
    # TODO: Maybe use kwargs for params later
    vision = False
    throttle = True
    gear_change = False
    # Agent only
    race_config_path = os.path.dirname(os.path.abspath(__file__)) + \
        "/raceconfig/agent_practice.xml"
    # Agent and one bot
    # race_config_path = os.path.dirname(os.path.abspath(__file__)) + \
    #     "/raceconfig/agent_damned_practice.xml"
    # Agent and 3 bots ?
    # race_config_path = os.path.dirname(os.path.abspath(__file__)) + \
    #     "/raceconfig/agent_damned_grid_practice.xml"

    # # Duh
    rendering = True
    lap_limiter = 2
    recdata = False
    rec_episode_limit = 300
    rec_timestep_limit = -1

    # env = gym.make(env_id)
    env = TorcsEnv(vision=vision, throttle=True, gear_change=False,
		race_config_path=race_config_path, rendering=rendering,
		lap_limiter = lap_limiter, recdata=recdata, noisy=True )

    # env = bench.Monitor(env, logger.get_dir() and os.path.join(logger.get_dir(), str(rank)))

    # TODO: Eval env incocation will kill Torcs bad
    # Either enable parallel torcs apps or kill training env and instantiate
    # eval env, then kill it again, and so on

    # Original
    # if evaluation and rank==0:
    #     eval_env = gym.make(env_id)
    #     eval_env = bench.Monitor(eval_env, os.path.join(logger.get_dir(), 'gym_eval'))
    #     env = bench.Monitor(env, None)
    # else:
    #     eval_env = None

    # Torcs Custom: the same env is used for eval ... fingercrossed
    # if evaluation and rank==0:
    #     eval_env = env
    #     eval_env = bench.Monitor(eval_env, os.path.join(logger.get_dir(), 'gym_eval'))
    #     env = bench.Monitor(env, None)
    # else:
    #     eval_env = None

    # Parse noise_type
    action_noise = None
    param_noise = None
    nb_actions = env.action_space.shape[-1]
    for current_noise_type in noise_type.split(','):
        current_noise_type = current_noise_type.strip()
        if current_noise_type == 'none':
            pass
        elif 'adaptive-param' in current_noise_type:
            _, stddev = current_noise_type.split('_')
            param_noise = AdaptiveParamNoiseSpec(initial_stddev=float(stddev), desired_action_stddev=float(stddev))
        elif 'normal' in current_noise_type:
            _, stddev = current_noise_type.split('_')
            action_noise = NormalActionNoise(mu=np.zeros(nb_actions), sigma=float(stddev) * np.ones(nb_actions))
        elif 'ou' in current_noise_type:
            _, stddev = current_noise_type.split('_')
            action_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(nb_actions), sigma=float(stddev) * np.ones(nb_actions))
        else:
            raise RuntimeError('unknown noise type "{}"'.format(current_noise_type))

    # Configure components.
    # TODO: Is memeory still needed ?
    memory = Memory(limit=int(1e6), action_shape=env.action_space.shape, observation_shape=env.observation_space.shape)
    # TODO: Is critic still needed ? probably not ...
    critic = Critic(layer_norm=layer_norm)
    actor = Actor(nb_actions, layer_norm=layer_norm)

    # Seed everything to make things reproducible.
    seed = seed + 1000000 * rank
    # logger.info('rank {}: seed={}, logdir={}'.format(rank, seed, logger.get_dir()))
    tf.reset_default_graph()
    set_global_seeds(seed)
    env.seed(seed)
    # if eval_env is not None:
    #     eval_env.seed(seed)

    # Disable logging for rank != 0 to avoid noise.
    if rank == 0:
        start_time = time.time()

    ### Importing Training code
    assert (np.abs(env.action_space.low) == env.action_space.high).all()  # we assume symmetric actions.
    max_action = env.action_space.high
    # logger.info('scaling actions by {} before executing in env'.format(max_action))
    agent = DDPG(actor, critic, memory, env.observation_space.shape, env.action_space.shape,
        gamma=gamma, tau=tau, normalize_returns=normalize_returns, normalize_observations=normalize_observations,
        batch_size=batch_size, action_noise=action_noise, param_noise=param_noise, critic_l2_reg=critic_l2_reg,
        actor_lr=actor_lr, critic_lr=critic_lr, enable_popart=popart, clip_norm=clip_norm,
        reward_scale=reward_scale)
    # logger.info('Using agent with the following configuration:')
    # logger.info(str(agent.__dict__.items()))

    # Set up logging stuff only for a single worker.
    if rank == 0:
        saver = tf.train.Saver()
    else:
        saver = None

    # Weight file name
    # save_filename = "/home/z3r0/random/rl/openai_logs/openai-ddpgtorcs-2018-08-21-22-34-42-448321/model_data/epoch_495.ckpt"
    # save_filename = "/home/z3r0/random/rl/openai_logs/openai-ddpgtorcs-2018-08-25-00-18-11-372623/model_data/epoch_495.ckpt"

    # 20180827 08-42-00-658308, Defiant, Training with enemy from the start, Best scoreing model DAMN This guy is good
    # save_filename = "/home/z3r0/random/rl/openai_logs/defiant/openai-ddpgtorcs-2018-09-03-13-10-09-537924/model_data/epoch_309.ckpt"

    # Alone special data collect trained agent
    # save_filename = "/home/z3r0/random/rl/openai_logs/defiant/openai-ddpgtorcs-2018-09-04-15-18-28-480417/model_data/epoch_104.ckpt"
    save_filename = "/home/z3r0/random/rl/openai_logs/defiant/openai-ddpgtorcs-2018-09-04-15-18-28-480417/model_data/epoch_742.ckpt"

    # openai-ddpgtorcs-2018-09-05-12-46-24-553500 Alone
    # save_filename = "/home/z3r0/random/rl/openai_logs/defiant/openai-ddpgtorcs-2018-09-05-12-46-24-553500/model_data/epoch_309.ckpt"
    # save_filename = "/home/z3r0/random/rl/openai_logs/defiant/openai-ddpgtorcs-2018-09-05-12-46-24-553500/model_data/epoch_104.ckpt"

    step = 0
    episode = 0
    eval_episode_rewards_history = deque(maxlen=100)
    episode_rewards_history = deque(maxlen=100)

    with U.single_threaded_session() as sess:
        # Prepare everything.
        agent.initialize(sess)
        sess.graph.finalize()

        agent.reset()
        obs = env.reset()

        # Restore to trained state ?
        saver.restore( sess, save_filename)

        done = False
        episode_reward = 0.
        episode_step = 0
        episodes = 0
        t = 0

        epoch = 0
        # Record variables
        expert_data = { "obs": [], "acs":[], "rews": [], "ep_rets": []}

        epoch_episode_rewards = []
        epoch_episode_steps = []
        epoch_episode_eval_rewards = []
        epoch_episode_eval_steps = []
        epoch_start_time = time.time()
        epoch_actions = []
        epoch_qs = []
        epoch_episodes = 0

        for episode in range(rec_episode_limit):
            done = False
            episode_reward = 0.
            t = 0
            start_time = time.time()

            ep_obs = []
            ep_rews = []
            ep_acs = []

            while not done:
                # Predict next action.
                # TODO: Noise on or off ?
                action, q = agent.pi(obs, apply_noise=False, compute_Q=True)

                assert action.shape == env.action_space.shape

                assert max_action.shape == action.shape
                new_obs, r, done, info = env.step(max_action * action)  # scale for execution in env (as far as DDPG is concerned, every action is in [-1, 1])
                t += 1

                episode_reward += r
                episode_step += 1

                # Book-keeping.
                # epoch_actions.append(action)
                # ep_acs.append(action)
                # epoch_qs.append(q)
                # agent.store_transition(obs, action, r, new_obs, done)
                ep_obs.append( obs)
                ep_rews.append( r)

                def clip_accel( accel):
                    if accel < 0.0:
                        return 0.0
                    else:
                        return accel

                ep_acs.append( [ action[0],
                    clip_accel(action[1])])

                obs = new_obs

                lapsed = (time.time() - start_time)

                # if  lapsed >= 30.0:
                #     done = True

                if done:
                    # Episode done.
                    # epoch_episode_rewards.append(episode_reward)
                    # episode_rewards_history.append(episode_reward)
                    # epoch_episode_steps.append(episode_step)

                    expert_data["obs"].append( ep_obs)
                    expert_data["acs"].append( ep_acs)
                    expert_data["rews"].append( ep_rews)
                    expert_data["ep_rets"].append( episode_reward)

                    episode_reward = 0.
                    episode_step = 0
                    epoch_episodes += 1
                    episodes += 1

                    agent.reset()

                    # Restore to trained state ?
                    saver.restore( sess, save_filename)
                    # Custom: Need to hard reset Torcs 'cause of mem leak
                    if np.mod(episodes, 5) == 0:
                        obs = env.reset(relaunch=True)   #relaunch TORCS every 3 episode because of the memory leak error
                    else:
                        obs = env.reset()
                    # obs = env.reset()
                    # print( len( obss))
                    # np.save( "/home/z3r0/torcs_data/ddpg_obs", np.asarray( obss))
                    # np.save( "/home/z3r0/torcs_data/ddpg_rews", np.asarray( rewss))
                    # np.save( "/home/z3r0/torcs_data/ddpg_acs", np.asarray( acss))
                    print( "Episode %d : TIme %.6f\n" % (episodes, lapsed))
                    # print( "Sampl. Rate: %f" % ( len( obss) / lapsed))
                    # print( "Episode reward %f" % ( np.sum( rewss)))

        # Save expert data
        np.savez( "/home/z3r0/torcs_data/ddpg_expert_data.npz", **expert_data)

    ### ENd training code

    env.close()
    # if eer.info('total runtime: {}s'.format(time.time() - start_time))

def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # parser.add_argument('--env-id', type=str, default='Pendulum-v0')
    # boolean_flag(parser, 'render-eval', default=False)
    boolean_flag(parser, 'layer-norm', default=True)
    boolean_flag(parser, 'render', default=False)
    boolean_flag(parser, 'normalize-returns', default=False)
    boolean_flag(parser, 'normalize-observations', default=False)
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--critic-l2-reg', type=float, default=1e-2)
    parser.add_argument('--batch-size', type=int, default=64)  # per MPI worker
    parser.add_argument('--actor-lr', type=float, default=1e-4)
    parser.add_argument('--critic-lr', type=float, default=1e-3)
    boolean_flag(parser, 'popart', default=False)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--reward-scale', type=float, default=1.)
    parser.add_argument('--clip-norm', type=float, default=None)
    parser.add_argument('--nb-epochs', type=int, default=1)  # with default settings, perform 1M steps total
    parser.add_argument('--nb-epoch-cycles', type=int, default=1)
    parser.add_argument('--nb-train-steps', type=int, default=10000000000)  # per epoch cycle and MPI worker
    parser.add_argument('--nb-eval-steps', type=int, default=100)  # per epoch cycle and MPI worker
    parser.add_argument('--nb-rollout-steps', type=int, default=10000000000)  # per epoch cycle and MPI worker
    parser.add_argument('--noise-type', type=str, default='adaptive-param_0.2')  # choices are adaptive-param_xx, ou_xx, normal_xx, none
    parser.add_argument('--num-timesteps', type=int, default=None)
    boolean_flag(parser, 'evaluation', default=False)
    args = parser.parse_args()
    # we don't directly specify timesteps for this script, so make sure that if we do specify them
    # they agree with the other parameters
    if args.num_timesteps is not None:
        assert(args.num_timesteps == args.nb_epochs * args.nb_epoch_cycles * args.nb_rollout_steps)
    dict_args = vars(args)
    del dict_args['num_timesteps']
    return dict_args

if __name__ == '__main__':
    args = parse_args()
    # DO not log
    # if MPI.COMM_WORLD.Get_rank() == 0:
        # logger.configure()
    # Run actual script.
    run(**args)
