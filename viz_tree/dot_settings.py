class DotSettings:
    def __init__(self, **kwargs):
        self.rankdir = 'TB'
        self.combine_lin = False
        self.use_predsplot = False
        self.use_regplot = False

        self.n_max = 5
        self.use_intercept = False
        self.fig_size = (5,3)
        self.feature_names = None
        self.display_type = "histogram"
        self.truncate_total_pred = False
        self.variable_tick_width = True
        self.highlight_x = None
        self.staircase = False

        # Override default values with inputs
        for key, value in kwargs.items():
            setattr(self, key, value)