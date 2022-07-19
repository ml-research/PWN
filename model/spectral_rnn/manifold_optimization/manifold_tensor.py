import torch


class ManifoldTensor(torch.Tensor):
    # Distance not computed on manifold, but rather just standard pytorch, i.e. euclidean

    # See https://github.com/pytorch/pytorch/issues/46159
    try:
        from torch._C import _disabled_torch_function_impl
        __torch_function__ = _disabled_torch_function_impl

    except ImportError:
        pass

    def __new__(cls, *args, manifold, requires_grad=False, **kwargs):
        data = args[0].data if len(args) == 1 and isinstance(args[0], torch.Tensor) else torch.Tensor(*args, **kwargs)

        if kwargs.get("device") is not None:
            data.data = data.data.to(kwargs.get("device"))

        instance = torch.Tensor._make_subclass(cls, data, requires_grad)
        instance.manifold = manifold

        return instance

    def __repr__(self):
        return f'Tensor on manifold {self.manifold} containing: {super().__repr__()}\n'

    # noinspection PyUnresolvedReferences
    def __reduce_ex__(self, proto):
        build, proto = super(ManifoldTensor, self).__reduce_ex__(proto)
        new_build = functools.partial(_rebuild_manifold_tensor, build_fn=build)
        new_proto = proto + (dict(), self.__class__, self.manifold, self.requires_grad)
        return new_build, new_proto
