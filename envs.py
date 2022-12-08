import gym
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from comet_ml import Experiment
import wandb
from utils import *

class PulseEnv(gym.Env):
    def __init__(self,
        trace_path,
        source_id,
        obs_ord=2,
        low=0.,
        high=5.,
        scale=1e8,
        seq_len=16,
        epi_len=1000,
        reward_func=DefaultL1,
        logger=None,
    ) -> None:
        super().__init__()
        self._trace_path = trace_path
        self._source_id = source_id
        self._obs_ord = obs_ord
        self._scale = scale
        self._seq_len = seq_len
        self._reward_func = reward_func
        self._epi_len = epi_len
        self._logger = logger

        self.action_space = gym.spaces.Box(np.array([low]), np.array([high]), dtype=np.float32)

        lows, highs = [low], [high]
        for _ in range(self._obs_ord-1):
            lows += [lows[-1] - highs[-1]]
            highs += [highs[-1] - lows[-1]]
        
        self.observation_space = gym.spaces.Box(
            np.vstack([np.array(lows)]*seq_len),
            np.vstack([np.array(highs)]*seq_len),
            dtype=np.float32
        )

        rece_times = []
        with open(self._trace_path, 'r') as f:
            log_items = f.readlines()
        for i, line in enumerate(tqdm(log_items, desc='initialize env - collecting log items')):
            # if i > 10000: break
            log_item = [int(x) for x in line.strip().split()]
            try:
                source_id, _, _, rece_time, _ = log_item
                if source_id != self._source_id:
                    continue
            except ValueError:
                ### erroneous log item (e.g. not enough items to unpack)
                continue
            rece_times.append(rece_time)
        
        if len(rece_times) == 0:
            raise KeyError(f'Current log file does not include pulses sent from node {self._source_id}')
        
        rece_times = np.array(rece_times) / self._scale
        self._gaps = rece_times[1:] - rece_times[:-1]
        del rece_times

    def reset(self):
        self.state = self._get_obs(np.zeros(self._seq_len))
        self._gap_idx = np.random.choice(self._gaps.shape[0]-self._epi_len)
        self._step_cnt = 0
        return self.state

    def step(self, action):
        ### Currently, the action does not affect the environment
        next_gap = self._gaps[self._gap_idx]
        
        cur_gaps = np.roll(self.state[:, 0], -1)
        cur_gaps[-1] = next_gap
        self.state = self._get_obs(cur_gaps)
        self._gap_idx += 1

        reward = self._reward_func(action, next_gap)

        self._step_cnt += 1
        done = bool(self._gap_idx >= len(self._gaps) or self._step_cnt == self._epi_len)

        info = {
            'step_reward': reward,
            'false_positive': int(action < next_gap),
            'detection_time': max(0, (action-next_gap).item()),
            'target': next_gap,
            'pred': action,
            # 'dis': dis,
            # 'rescaled_dis': dis * self._scale
        }
        if self._logger:
            if isinstance(self._logger, Experiment):
                self._logger.log_metrics(info, step=1)
            else:  # wandb
                self._logger.log(info)

        return self.state, reward, done, info

    def render(self, mode='human', close=False):
        pass

    def _get_obs(self, gaps):
        ### e.g. when obs_ord = 3 
        ### 1, 5, 3, 5, 9
        ### 0, 4, -2, 2, 4
        ### 0, 0, -6, 4, 2
        
        obs = np.zeros((gaps.shape[0], self._obs_ord))
        obs[:, 0] = gaps
        for i in range(self._obs_ord-1):
            obs[i+1:, i+1] = obs[i+1:, i] - obs[:-i-1, i]
        assert obs.shape == (self._seq_len, self._obs_ord)
        return obs

"""
if __name__ == '__main__':
    
    trace_path = './traces/trace.log'
    node_id = 5
    env = PulseEnv(trace_path, node_id, obs_ord=3, seq_len=64)
    
    o = env.reset()
    d = None
    r = 0.
    while not d:
        # print(o[:10])
        a = env.action_space.sample()
        o, r, d, _ = env.step(a)
        print(env._gap_idx)
"""