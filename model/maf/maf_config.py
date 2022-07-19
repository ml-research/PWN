class MAFConfig:
    def __init__(self, use_limited_context=None, prepare_joint=False, window_level=False):

        self.use_limited_context = use_limited_context
        self.prepare_joint = prepare_joint
        self.window_level = window_level
