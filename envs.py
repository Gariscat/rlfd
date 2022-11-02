import gym
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


class PulseEnv(gym.Env):
    def __init__(self,
        trace_path,
        source_id,
        obs_ord=2,
        low=0.,
        high=5.,
        scale=1e8,
        num_gaps=16,
        reward_func=None,
    ) -> None:
        super().__init__()
        self._trace_path = trace_path
        self._source_id = source_id
        self._obs_ord = obs_ord
        self._scale = scale
        self._num_gaps = num_gaps
        self._reward_func = reward_func

        self.action_space = gym.spaces.Box(np.array([low]), np.array([high]), dtype=np.float64)

        lows, highs = [low], [high]
        for _ in range(self._obs_ord-1):
            lows += [lows[-1] - highs[-1]]
            highs += [highs[-1] - lows[-1]]
        
        self.observation_space = gym.spaces.Box(
            np.vstack([np.array(lows)]*num_gaps),
            np.vstack([np.array(highs)]*num_gaps),
            dtype=np.float64
        )

        rece_times = []
        with open(self._trace_path, 'r') as f:
            log_items = f.readlines()
        for line in tqdm(log_items, desc='initialize env - collecting log items'):
            log_item = [int(x) for x in line.strip().split()]
            try:
                source_id, _, _, rece_time, _ = log_item
                if source_id != self._source_id:
                    continue
            except ValueError:
                continue # erroneous log item (e.g. not enough items to unpack)
            rece_times.append(rece_time)
        
        if len(rece_times) == 0:
            raise KeyError(f'Current log file does not include pulses sent from node {self._source_id}')
        
        rece_times = np.array(rece_times) / self._scale
        self._gaps = rece_times[1:] - rece_times[:-1]
        del rece_times

    def reset(self):
        self.state = self._get_obs(np.zeros(self._num_gaps))
        self._gap_idx = 0
        return self.state

    def step(self, action):
        ### Currently, the action does not affect the environment
        next_gap = self._gaps[self._gap_idx]
        
        cur_gaps = np.roll(self.state[:, 0], -1)
        cur_gaps[-1] = next_gap
        self.state = self._get_obs(cur_gaps)
        self._gap_idx += 1

        if self._reward_func is None:
            # default is e^(-dis)
            reward = np.exp(-np.abs(action-next_gap))
        else:
            reward = self._reward_func(action, next_gap)

        done = bool(self._gap_idx >= len(self._gaps))

        return self.state, reward, done, {}

    def _get_obs(self, gaps):
        ### e.g. when obs_ord = 3 
        ### 1, 5, 3, 5, 9
        ### 0, 4, -2, 2, 4
        ### 0, 0, -6, 4, 2
        
        obs = np.zeros((gaps.shape[0], self._obs_ord))
        obs[:, 0] = gaps
        for i in range(self._obs_ord-1):
            obs[i+1:, i+1] = obs[i+1:, i] - obs[:-i-1, i]
        assert obs.shape == (self._num_gaps, self._obs_ord)
        return obs


if __name__ == '__main__':
    """
    trace_path = './traces/trace.log'
    node_id = 5
    env = PulseEnv(trace_path, node_id, obs_ord=3, num_gaps=64)
    
    o = env.reset()
    d = None
    r = 0.
    while not d:
        # print(o[:10])
        a = env.action_space.sample()
        o, r, d, _ = env.step(a)
        print(env._gap_idx)
    """