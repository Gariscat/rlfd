from envs import PulseEnv
from networks import RNNEncoder
import stable_baselines3
from stable_baselines3 import DDPG

env = PulseEnv('./traces/trace.log', source_id=3, obs_ord=3, epi_steps=10000, num_gaps=128)
# stable_baselines3.common.env_checker.check_env(env)

policy_kwargs = dict(
    features_extractor_class=RNNEncoder,
    features_extractor_kwargs=dict(features_dim=32, num_layers=3),
)
model = DDPG('MlpPolicy', env, buffer_size=int(1e5), tensorboard_log='./logs/', policy_kwargs=policy_kwargs, verbose=2)
model.learn(total_timesteps=5_000_000, log_interval=1, progress_bar=True)

for _ in range(10):
    obs = env.reset()
    done = False
    while True:
        action, _states = model.predict(obs)
        obs, reward, done, info = env.step(action)
        # print(action, rewards)
        if done:
            break