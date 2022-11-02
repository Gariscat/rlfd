import gym
import stable_baselines3

import torch
from torch import nn

class RNNEncoder(stable_baselines3.common.torch_layers.BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """
    def __init__(self,
        observation_space: gym.spaces.Box,
        features_dim: int = 32,
        hidden_size: int = 32,
        # seq_len: int = 256,
        num_layers: int = 3,
        rnn: nn.Module = nn.LSTM
    ):
        super().__init__(observation_space, features_dim)
        assert features_dim == hidden_size

        self.net = rnn(
            input_size=observation_space.shape[-1],
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )

    def forward(self, observations):
        # print(type(observations), observations.shape, type(self.net))
        last_hidden_states, (_, _) = self.net(observations)
        # print(last_hidden_states.shape)
        preds = last_hidden_states[:, -1, :]
        return preds
