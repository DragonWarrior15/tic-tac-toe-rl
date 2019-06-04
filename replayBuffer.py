from collections import deque
import numpy as np

# a class to keep snapshots for training
class ReplayBuffer():
    # init
    def __init__(self, buffer_size = 100):
        self._buffer = deque(maxlen = buffer_size)

    def add_data(self, data):
        self._buffer.append(data)

    def sample(self, size = None):
        buffer_size = len(self._buffer)
        size = min(size, buffer_size)
        p = (1.0*size)/buffer_size
        # sample size will be smaller than buffer size, hence traverse queue once
        sample_data = []
        for x in self._buffer:
            if(np.random.random() < p):
                sample_data.append(x)
        np.random.shuffle(sample_data)
        s, a, r, next_s, done = [], [], [], [], []
        for x in sample_data:
            s.append(x[0])
            a.append(x[1])
            r.append(x[2])
            next_s.append(x[3])
            done.append(x[4])
        s = np.concatenate(s)
        a = np.concatenate(a, axis=0)
        r = np.array(r).reshape(-1, 1)
        next_s = np.concatenate(next_s)
        done = np.array(done).reshape(-1, 1)

        return s, a, r, next_s, done
