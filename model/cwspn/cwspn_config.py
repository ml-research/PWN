class CWSPNConfig:
    # Sigma interval of (0.01, 4) corresponds to a std interval of (0.1, 2)
    def __init__(self, use_limited_context=None, rg_splits=8, rg_split_size=2, rg_split_recursion=2, num_sums=5,
                 num_gauss=4, gauss_min_mean=None,  gauss_max_mean=None, gauss_min_sigma=0.01, gauss_max_sigma=1,
                 use_rationals=False):

        # For WCSPN, both joint preparation & window_level can't be used
        self.prepare_joint = False
        self.window_level = False

        self.use_limited_context = use_limited_context
        self.rg_splits = rg_splits
        self.rg_split_size = rg_split_size
        self.rg_split_recursion = rg_split_recursion
        self.num_sums = num_sums
        self.num_gauss = num_gauss
        self.gauss_min_mean = gauss_min_mean
        self.gauss_max_mean = gauss_max_mean
        self.gauss_min_sigma = gauss_min_sigma
        self.gauss_max_sigma = gauss_max_sigma
        self.use_rationals = use_rationals
