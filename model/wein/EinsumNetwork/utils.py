import torch


def one_hot(x, K, dtype=torch.float):
    """One hot encoding"""
    with torch.no_grad():
        ind = torch.zeros(x.shape + (K,), dtype=dtype, device=x.device)
        ind.scatter_(-1, x.unsqueeze(-1), 1)
        return ind


def sample_matrix_categorical(p):
    """Sample many Categorical distributions represented as rows in a matrix."""
    with torch.no_grad():
        cp = torch.cumsum(p[:, 0:-1], -1)
        rand = torch.rand((cp.shape[0], 1), device=cp.device)
        rand_idx = torch.sum(rand > cp, -1).long()
        return rand_idx
