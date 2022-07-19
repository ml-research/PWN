class RNNLayerConfig:
    def __init__(self, use_cg_cell=False, use_gated=True, use_linear_projection=True, use_residual=True,
                 n_layers=1, learn_hidden_init=False, dropout=None):
        self.use_cg_cell = use_cg_cell
        self.use_gated = use_gated
        self.use_linear_projection = use_linear_projection
        self.use_residual = use_residual
        self.learn_hidden_init = learn_hidden_init
        self.n_layers = n_layers
        self.dropout = dropout


class SpectralRNNConfig:
    # Note: This class serves only as a typed dict - therefore, no checks for attribute integrity are performed
    # As a result, if e.g. changing only window_size, step_width = int(window_size * overlap) may not hold anymore

    def __init__(self, window_size=96, overlap=0.5, fft_compression=4, normalize_fft=False,
                 embedding_dim=32, hidden_dim=64, clip_gradient_value=1, use_add_linear=False, x_as_labels=False,
                 use_only_ts_input=True, rnn_layer_config: RNNLayerConfig = RNNLayerConfig(),
                 use_cached_predictions=False):
        # Set fft_compression to 1 for no compression at all
        # Set clip_gradient_value to <= 0 to deactivate

        self.window_size = window_size
        self.overlap = overlap
        self.fft_compression = fft_compression
        self.normalize_fft = normalize_fft
        self.embedding_dim = embedding_dim
        self.clip_gradient_value = clip_gradient_value
        self.hidden_dim = hidden_dim
        self.use_add_linear = use_add_linear
        self.x_as_labels = x_as_labels
        self.use_only_ts_input = use_only_ts_input
        self.rnn_layer_config = rnn_layer_config
        self.use_cached_predictions = use_cached_predictions

        self.step_width = None
        self.input_dim = None
        self.value_dim = None
        self.compressed_value_dim = None
        self.removed_freqs = None
        self.embedding_sizes = None
