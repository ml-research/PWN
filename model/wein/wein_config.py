from .EinsumNetwork import EinsumNetwork


class WEinConfig:
    def __init__(self, window_level=False, prepare_joint=False, use_limited_context=None, K=1,
                 structure={'type': 'binary-trees', 'depth': 4, 'num_repetitions': 7},
                 exponential_family=EinsumNetwork.MultivariateNormalArray,
                 exponential_family_args={'min_var': 1e-6, 'max_var': 0.01}, online_em_frequency=1,
                 online_em_stepsize=0.05):
        self.window_level = window_level
        self.prepare_joint = prepare_joint
        self.use_limited_context = use_limited_context
        self.K = K
        self.structure = structure
        self.exponential_family = exponential_family
        self.exponential_family_args = exponential_family_args
        self.online_em_frequency = online_em_frequency
        self.online_em_stepsize = online_em_stepsize

        self.input_size = None
