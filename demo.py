from envs import PulseEnv
from networks import RNNEncoder

from stable_baselines3 import DDPG

env = PulseEnv('./traces/trace.log', source_id=3, obs_ord=3, num_gaps=256)

policy_kwargs = dict(
    features_extractor_class=RNNEncoder,
    features_extractor_kwargs=dict(features_dim=32, num_layers=1),
)
model = DDPG('MlpPolicy', env, buffer_size=int(1e5), policy_kwargs=policy_kwargs, verbose=2)
# model.learn(100, log_interval=10)
obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    print(action, rewards)
    if dones:
        break