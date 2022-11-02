from envs import PulseEnv
from networks import RNNEncoder

from stable_baselines3 import DDPG

env = PulseEnv('./traces/trace.log', source_id=3, obs_ord=3, num_gaps=256)

policy_kwargs = dict(
    features_extractor_class=RNNEncoder,
    features_extractor_kwargs=dict(features_dim=32),
)
model = DDPG('MlpPolicy', env, buffer_size=int(1e5), policy_kwargs=policy_kwargs, verbose=1)
model.learn(int(1e6))