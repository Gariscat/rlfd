import os
from datetime import datetime
import re
from envs import PulseEnv
from networks import RNNEncoder
import stable_baselines3
from stable_baselines3 import DDPG
from utils import *
import json
import argparse


parser = argparse.ArgumentParser()

parser.add_argument('--trace_path', type=str, default='./traces/trace.log', help='The path of the log traces.')
parser.add_argument('--source_id', type=int, default=3, help='The node id of the message sender.')
parser.add_argument('--obs_ord', type=int, default=3, help='The highest order of observations.')
parser.add_argument('--epi_len', type=int, default=10000, help='Length of an episode.')
parser.add_argument('--seq_len', type=int, default=128, help='Length of the input of RNN.')
parser.add_argument('--features_dim', type=int, default=32, help='Dim of the output of RNN.')
parser.add_argument('--reward_func', type=str, default='DefaultL1', help='Reward function of the environment.')
parser.add_argument('--num_layers', type=int, default=3, help='Number of layers of the RNN.')
parser.add_argument('--buffer_size', type=int, default=100_000, help='Size of the replay buffer.')
parser.add_argument('--run_dir', type=str, default='./runs/', help='Logger directory.')
parser.add_argument('--ckpt_dir', type=str, default='./checkpoints/', help='Checkpoint directory.')
parser.add_argument('--tot_steps', type=int, default=200_000, help='Total number of training steps.')
parser.add_argument('--seed', type=int, default=0, help='Random seed.')
# args.DEFINE_bool('debug', False, 'Debug mode does not store training logs.')
parser.add_argument('--logger', type=str, default='wandb', help='Which 3rd party logger to use.')
args = parser.parse_args()


if __name__ == '__main__':
    os.makedirs(args.run_dir, exist_ok=True)
    os.makedirs(args.ckpt_dir, exist_ok=True)
    
    config_dict = {
        "obs_ord": args.obs_ord,
        "seq_len": args.seq_len,
        "features_dim": args.features_dim,
        "num_layers": args.num_layers,
        "tot_steps": args.tot_steps,
        'reward_func': args.reward_func
    }
    
    logger = None
    if args.logger == 'comet':
        from comet_ml import Experiment
        logger = Experiment(
            api_key='BZy4qcxu4uysjvaOdgTjCfT2n',
            project_name='rlfd',
            workspace='gariscat',
        )
        logger.log_parameters(config_dict)
    elif args.logger == 'wandb':
        import wandb
        wandb.init(project='rlfd', entity='kgv007', config=config_dict)
        logger = wandb

    reward_func = DefaultL1
    if args.reward_func == 'L1ConstPunishFP':
        reward_func = L1ConstPunishFP
    elif args.reward_func == 'L1TanhPunishFP':
        reward_func = L1TanhPunishFP

    env = PulseEnv(
        trace_path=args.trace_path,
        source_id=args.source_id,
        obs_ord=args.obs_ord,
        epi_len=args.epi_len,
        seq_len=args.seq_len,
        reward_func=reward_func,
        logger=logger,
    )
    env.seed(args.seed)
    # stable_baselines3.common.env_checker.check_env(env)

    policy_kwargs = dict(
        features_extractor_class=RNNEncoder,
        features_extractor_kwargs=dict(features_dim=args.features_dim, num_layers=args.num_layers),
    )
    model = DDPG(
        policy='MlpPolicy',
        env=env,
        buffer_size=args.buffer_size,
        tensorboard_log=args.run_dir,
        policy_kwargs=policy_kwargs,
        seed=args.seed,
        verbose=2,
    )
    
    config_str = f'ord_{args.obs_ord}_seqlen_{args.seq_len}_dim_{args.features_dim}_layers_{args.num_layers}_reward_{args.reward_func}'
    model.learn(
        total_timesteps=args.tot_steps,
        log_interval=1,
        tb_log_name=config_str,
    )
    ckpt_name = re.sub(r'\W+', '', str(datetime.now()))
    model.save(os.path.join(args.ckpt_dir, config_str+'_'+ckpt_name))

    eval_results = evaluate(env, model, num_episodes=10)
    eval_results.update(config_dict)
    print(eval_results)
    with open(os.path.join(args.run_dir, config_str)+'_1/eval_results.json', 'w') as f:
        json.dump(eval_results, f)
