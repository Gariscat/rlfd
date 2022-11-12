from absl import flags, app
import os
from datetime import datetime
import re

from envs import PulseEnv
from networks import RNNEncoder
import stable_baselines3
from stable_baselines3 import DDPG
import wandb


FLAGS = flags.FLAGS
flags.DEFINE_string('trace_path', './traces/trace.log', 'The path of the log traces.')
flags.DEFINE_integer('source_id', 3, 'The node id of the message sender.')
flags.DEFINE_integer('obs_ord', 3, 'The highest order of observations.')
flags.DEFINE_integer('epi_len', 10000, 'Length of an episode.')
flags.DEFINE_integer('seq_len', 128, 'Length of the input of RNN.')
flags.DEFINE_integer('features_dim', 32, 'Dim of the output of RNN.')
flags.DEFINE_integer('num_layers', 3, 'Number of layers of the RNN.')
flags.DEFINE_integer('buffer_size', 100_000, 'Size of the replay buffer.')
flags.DEFINE_string('run_dir', './runs/', 'Logger directory.')
flags.DEFINE_string('ckpt_dir', './checkpoints/', 'Checkpoint directory.')
flags.DEFINE_integer('tot_steps', 5_000_000, 'Total number of training steps.')
flags.DEFINE_integer('seed', 0, 'Random seed.')


def main(_):
    os.makedirs(FLAGS.run_dir, exist_ok=True)
    os.makedirs(FLAGS.ckpt_dir, exist_ok=True)
    
    hyper = {
        "obs_ord": FLAGS.obs_ord,
        "seq_len": FLAGS.seq_len,
        "features_dim": FLAGS.features_dim,
        "num_layers": FLAGS.num_layers,
        "tot_steps": FLAGS.tot_steps
    }
    wandb.init(project="rlfd-debug", entity='gariscat', config=hyper)
    
    env = PulseEnv(
        trace_path=FLAGS.trace_path,
        source_id=FLAGS.source_id,
        obs_ord=FLAGS.obs_ord,
        epi_len=FLAGS.epi_len,
        seq_len=FLAGS.seq_len,
        logger=wandb,
    )
    env.seed(FLAGS.seed)
    # stable_baselines3.common.env_checker.check_env(env)

    policy_kwargs = dict(
        features_extractor_class=RNNEncoder,
        features_extractor_kwargs=dict(features_dim=FLAGS.features_dim, num_layers=FLAGS.num_layers),
    )
    model = DDPG(
        policy='MlpPolicy',
        env=env,
        buffer_size=FLAGS.buffer_size,
        tensorboard_log=FLAGS.run_dir,
        policy_kwargs=policy_kwargs,
        seed=FLAGS.seed,
        verbose=2,
    )
    
    config_info = f'ord_{FLAGS.obs_ord}_seqlen_{FLAGS.seq_len}_dim_{FLAGS.features_dim}_layers_{FLAGS.num_layers}'
    model.learn(
        total_timesteps=FLAGS.tot_steps,
        log_interval=1,
        tb_log_name=config_info,
        progress_bar=True
    )
    ckpt_name = re.sub(r'\W+', '', str(datetime.now()))
    model.save(os.path.join(FLAGS.ckpt_dir, config_info+'_'+ckpt_name))
    """"""
    """
    for _ in range(10):
        obs = env.reset()
        done = False
        while True:
            action, _states = model.predict(obs)
            obs, reward, done, info = env.step(action)
            # print(action, rewards)
            if done:
                break
    """

if __name__ == '__main__':
    app.run(main)