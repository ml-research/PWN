import torch
import numpy as np


def to_complex_activation(activation):
    return lambda x: torch.view_as_complex(torch.cat(
        [activation(x.real).unsqueeze(-1), activation(x.imag).unsqueeze(-1)], dim=-1))


def mod_relu_act(offset):
    """
    Arjovsky et al. Unitary Evolution Recurrent Neural Networks
    https://arxiv.org/abs/1511.06464
    """

    def mod_relu(z):
        modulus = torch.sqrt(z.real ** 2 + z.imag ** 2)
        rescale = torch.relu(modulus + offset) / (modulus + 1e-6)

        return torch.complex(rescale, torch.zeros_like(rescale)) * z

    return mod_relu


def rand_uniform(shape, lower, upper, dtype=torch.float):
    return (lower - upper) * torch.rand(*shape, dtype=dtype) + lower


def clip_grad_value_complex_(parameters, clip_value):
    """
    Taken and modified from https://pytorch.org/docs/stable/generated/torch.clamp.html
    """

    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]

    clip_value = float(clip_value)
    for p in filter(lambda p: p.grad is not None, parameters):
        if p.dtype in [torch.complex32, torch.complex64, torch.complex128]:
            p.grad.data.real.clamp_(min=-clip_value, max=clip_value)
            p.grad.data.imag.clamp_(min=-clip_value, max=clip_value)
        else:
            p.grad.data.clamp_(min=-clip_value, max=clip_value)


def unitary_init(shape):
    """
    Taken and modified from https://github.com/v0lta/Complex-gated-recurrent-neural-networks/
    """

    limit = np.sqrt(6 / (shape[0] + shape[1]))
    rand_r = np.random.uniform(-limit, limit, shape[0:2])
    rand_i = np.random.uniform(-limit, limit, shape[0:2])
    crand = rand_r + 1j * rand_i
    u, s, vh = np.linalg.svd(crand)
    # use u and vg to create a unitary matrix:
    unitary = np.matmul(u, np.transpose(np.conj(vh)))

    # test_eye = np.matmul(np.transpose(np.conj(unitary)), unitary)
    # print('I - Wi.H Wi', np.linalg.norm(test_eye) - unitary)
    # # test
    #
    # import matplotlib.pyplot as plt
    # plt.imshow(np.abs(np.matmul(unitary, np.transpose(np.conj(unitary))))); plt.show()
    stacked = np.stack([np.real(unitary), np.imag(unitary)], -1)
    assert stacked.shape[:2] == tuple(shape), "Unitary initialization shape mismatch."

    return torch.view_as_complex(torch.tensor(stacked, dtype=torch.float))
