import torch

def set_timestamps(timestamps):
    return [timestamps[i].unique() for i in range(len(timestamps))]


def get_indices(timestamps, set_ts, bs):
    all_idx = []
    for i in range(len(set_ts)):
        idx = []
        for j in range(bs):
            idx.append((set_ts[i] == timestamps[i][j]).nonzero().item())
        all_idx.append(idx)
    return all_idx