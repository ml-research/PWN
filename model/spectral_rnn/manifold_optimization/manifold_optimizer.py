import torch

from .manifold import Manifold
from .manifold_parameter import ManifoldParameter


# Applies RMSProp to manifold parameters
class ManifoldOptimizer(torch.optim.Optimizer):

    def __init__(self, params, lr, base_optimizer, alpha=0.99, **kwargs):
        # Separate manifold parameters from standard parameters
        base_params = []
        manifold_params = []
        for param in params:
            if isinstance(param, ManifoldParameter):
                manifold_params.append(param)
            else:
                base_params.append(param)

        super(ManifoldOptimizer, self).__init__(manifold_params, {'lr': lr})

        self.lr = lr
        self.alpha = alpha
        self.base_optimizer = base_optimizer(base_params, lr, alpha=alpha, *kwargs)

    @torch.no_grad()
    def step(self, closure=None):
        """
        Performs a single optimization step on the manifold parameters and calls the base_optimizer to perform a step
            on the remaining parameters.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        self.base_optimizer.step(None)

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                # They dont do grad scaling as normally done in RMSProp
                if p.manifold == Manifold.STIEFEL:
                    # What about the Sherman-Morrison-Woodbury Formula?
                    grad = p.grad
                    state = self.state[p]

                    # State initialization
                    if len(state) == 0:
                        state['step'] = 0
                        state['square_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                    state['step'] += 1
                    state['square_avg'] = state['square_avg'] * self.alpha + (1 - self.alpha) * grad * grad

                    grad_rescaled = grad / state['square_avg']

                    eye = torch.eye(*grad_rescaled.shape).to(grad_rescaled.device)
                    a = p @ grad_rescaled.conj().T - p.conj().T @ grad_rescaled
                    p -= torch.inverse((eye + (self.lr / 2.0) * a)) @ (eye - (self.lr / 2.0) * a) @ p

                else:
                    raise ValueError(f'Unsupported manifold type {p.manifold}')

        return loss
