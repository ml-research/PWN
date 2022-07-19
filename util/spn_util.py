from model.cwspn import CWSPN

import torch
import numpy as np


def get_fc_confidence(wcspn: WCSPN, x_, y_):
    x, y = wcspn.prepare_input(x_, y_, stft_y=False)

    sum_params, leaf_params = wcspn.weight_nn(x.reshape((x.shape[0], x.shape[1] * 2)))
    wcspn.args.param_provider.sum_params = sum_params.detach()
    wcspn.args.param_provider.leaf_params = leaf_params.detach()

    means = [[] for _ in range(y.shape[1])]
    sigmas = [[] for _ in range(y.shape[1])]
    for leaf_vector in wcspn.spn.vector_list[0]:
        for i, rv in enumerate(leaf_vector.scope):
            means[rv].append(leaf_vector.get_means()[0, i])
            sigmas[rv].append(leaf_vector.get_sigma()[0, i])

    final_means = np.zeros((y_.shape[1], 2))
    final_sigmas = np.zeros((y_.shape[1], 2))
    for rv, means_, sigmas_ in zip(range(y.shape[1]), means, sigmas):
        final_means[rv] = torch.stack(means_, dim=0).mean(dim=0).mean(dim=0).detach().cpu().numpy()
        final_sigmas[rv] = torch.sqrt(torch.stack(sigmas_, dim=0).mean(dim=0).mean(dim=0)).detach().cpu().numpy()

    return final_means, final_sigmas, wcspn.spn(y)


def get_fc_grads(wcspn: WCSPN, x_, y_):
    x, y = wcspn.prepare_input(x_, y_, stft_y=False)
    criterion = lambda out: torch.logsumexp(out, dim=1).mean()

    y = torch.autograd.Variable(torch.complex(y[..., 0], y[..., 1]), requires_grad=True)

    sum_params, leaf_params = wcspn.weight_nn(x.reshape((x.shape[0], x.shape[1] * 2)))
    wcspn.args.param_provider.sum_params = sum_params.detach()
    wcspn.args.param_provider.leaf_params = leaf_params.detach()

    def call_wcspn(y_in):
        out = wcspn.spn(torch.stack([y_in.real, y_in.imag], dim=-1))
        return criterion(out)

    grads_ = torch.autograd.functional.hessian(call_wcspn, y).detach().cpu().numpy()  # Estimate of fisher information matrix
    # grads_ = torch.autograd.functional.jacobian(call_wcspn, y).detach().cpu().numpy()

    grads = grads_[:, range(grads_.shape[1]), 0, range(grads_.shape[-1])]

    return grads, wcspn.spn(torch.stack([y.real, y.imag], dim=-1))
