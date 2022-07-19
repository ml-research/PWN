import torch

from .manifold_tensor import ManifoldTensor


class ManifoldParameter(torch.nn.Parameter):

    def __new__(cls, data, manifold=None, requires_grad=True):
        if not isinstance(data, ManifoldTensor):
            data = ManifoldTensor(data, manifold=manifold)
        else:
            assert manifold is None or data.manifold == manifold

        instance = ManifoldTensor._make_subclass(cls, data, requires_grad)
        instance.manifold = data.manifold

        return instance

    def __repr__(self):
        return f'Parameter on manifold {self.manifold} containing: {super().__repr__()}\n'
