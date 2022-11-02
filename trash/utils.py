import torch
from torch.utils.data import Dataset, ConcatDataset
import numpy as np


class SingleSourceSingleSinkDataset(Dataset):
    def __init__(self, source_id, logs, d, scale) -> None:
        super().__init__()
        # logs = np.array(logs)
        # self.source_id = torch.tensor(logs[:, 0], dtype=torch.long)
        # self.pulse_id = torch.tensor(logs[:, 1], dtype=torch.long)
        # self.e_time = torch.tensor(logs[:, 2], dtype=torch.float) / scale
        self._source_id = source_id
        self._length = len(logs)
        self._receive_times = torch.tensor(logs[:, 3], dtype=torch.float) / scale
        # self.num_hops = torch.tensor(logs[:, 4], dtype=torch.int)
        self._inputs = torch.zeros((self._length, d+1))
        self._inputs[:, 0] = self._receive_times

        for i in range(1, d+1):
            self._inputs[i:, i] = self._inputs[i-1:, i-1] - self._inputs[:-i, i-1]
        
        self._targets = torch.zeros(self._length)
        self._targets[:-1] = self._length[1:, 1]
        

    def __len__(self):
        return self._length

    def __item__(self, idx):
        return self._source_id, self._inputs[idx], self._targets[idx]


def MultiSourceSingleSinkDataset(data_path, d=2, scale=1e8, K=256, only_receive_time=True) -> None:
    with open(data_path, 'r') as f:
        lines = f.readlines()
    for i in range(len(lines)):
        temp = lines[i].strip().split()
        lines[i] = [int(x) for x in temp]
    groups = {}
    for log_item in lines:
        source_id, pulse_id, emit_time, receive_time, num_hops = log_item
        if source_id not in groups.keys():
            groups[source_id] = []
        if only_receive_time:
            groups[source_id].append(receive_time)
        else: # not implemented
            pass

    single_source_data = [SingleSourceSingleSinkDataset(k, groups[k], d, scale) for k in groups.keys()]
    return ConcatDataset(single_source_data)


if __name__ == '__main__':
    trace_data = MultiSourceSingleSinkDataset('./traces/trace.log')
    print(len(trace_data))