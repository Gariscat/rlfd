import gym
import numpy as np


class PulseEnv(gym.Env):
    def __init__(self,
        trace_path,
        source_id,
        obs_ord=2,
        low=0.,
        high=5.,
        scale=1e8,
        num_gaps=256,
        reward_func=None,
    ) -> None:
        super().__init__()
        self._trace_path = trace_path
        self._source_id = source_id
        self._scale = scale
        self._num_gaps = num_gaps
        self._reward_func = reward_func

        self.action_space = gym.spaces.Box(low, high, dtype=np.float64)

        lows, highs = [low], [high]
        for _ in range(obs_ord-1):
            lows += [lows[-1] - highs[-1]]
            highs += [highs[-1] - lows[-1]]
        # print(lows, highs)
        self.observation_space = gym.spaces.Box(np.array(lows), np.array(highs), dtype=np.float64)

    def reset(self):
        rece_times = []
        with open(self._trace_path, 'r') as f:
            log_items = f.readlines()
        for line in log_items:
            log_item = [int(x) for x in line.strip().split()]
            source_id, _, _, rece_time, _ = log_item
            if source_id != self._source_id:
                continue
            rece_times.append(rece_time)
        
        if len(rece_times) == 0:
            raise KeyError(f'Current log file does not include pulses sent from node {self._source_id}')
        
        rece_times = np.array(rece_times) / self._scale
        self._gaps = rece_times[1:] - rece_times[:-1]

        self.state = np.zeros(self._num_gaps)
        self._gap_idx = 0
        return self.state

    def step(self, action):
        ### Currently, the action does not affect the environment
        next_gap = self._gaps[self._gap_idx]
        
        self.state = np.roll(self.state, -1)
        self.state[-1] = next_gap
        self._gap_idx += 1

        if self._reward_func is None:
            # default is e^(-dis)
            reward = np.exp(-np.abs(action-next_gap))
        else:
            reward = self._reward_func(action, next_gap)

        done = bool(self._gap_idx >= len(self._gaps))

        return self.state, reward, done, {}


if __name__ == '__main__':
    """
    trace_path = './traces/trace.log'
    node_id = 5
    env = PulseEnv(trace_path, node_id, obs_ord=3)
    print(env.observation_space)
    """