class TransformerConfig:
    # Note: This class serves only as a typed dict - therefore, no checks for attribute integrity are performed
    # As a result, if e.g. changing only window_size, step_width = int(window_size * overlap) may not hold anymore

    def __init__(self, window_size=96, overlap=0.5, fft_compression=4, normalize_fft=False,
                 embedding_dim=32, hidden_dim=64, clip_gradient_value=1, use_add_linear=False, x_as_labels=False,
                 use_only_ts_input=True, use_cached_predictions=False, is_complex=True, native_complex=True,
                 attention_size=None, dropout=0.05, heads=8, q=8, k=8, num_enc_dec=1, chunk_mode=None,
                 pe='original'):
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
        self.use_cached_predictions = use_cached_predictions

        self.is_complex = is_complex
        self.native_complex = native_complex
        self.attention_size = attention_size
        self.dropout = dropout
        self.heads = heads
        self.q = q
        self.k = k
        self.num_enc_dec = num_enc_dec
        self.chunk_mode = chunk_mode
        self.pe = pe

        self.step_width = None
        self.input_dim = None
        self.value_dim = None
        self.compressed_value_dim = None
        self.removed_freqs = None
        self.embedding_sizes = None
