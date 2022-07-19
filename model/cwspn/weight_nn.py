import torch.nn as nn
from rational.torch import Rational


class WeigthNN(nn.Module):

    def __init__(self, inp_size, num_sum_params, num_leaf_params, use_rationals=False):
        super(WeigthNN, self).__init__()

        self.num_sum_params = num_sum_params
        self.num_leaf_params = num_leaf_params

        self.sums_roundup = ((num_sum_params + 63) // 64) * 64
        self.dense_s1 = nn.Linear(inp_size, self.sums_roundup // 2)
        self.dense_sa = Rational() if use_rationals else nn.ReLU()
        self.dense_s2 = nn.Linear(8 * 4, 64)

        hidden_l = num_leaf_params // 4
        self.dense_l1 = nn.Linear(inp_size, hidden_l)
        self.dense_la = Rational() if use_rationals else nn.ReLU()
        self.dense_l2 = nn.Linear(hidden_l, num_leaf_params)

    def forward(self, x):
        sum_ = self.dense_sa(self.dense_s1(x)).reshape((-1, max(self.sums_roundup // 64 // 4, 1), 8 * 4))
        leaf_ = self.dense_la(self.dense_l1(x))

        return self.dense_s2(sum_).reshape((-1, self.sums_roundup)), self.dense_l2(leaf_)