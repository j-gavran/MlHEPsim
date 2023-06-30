class BasePipeline:
    def __init__(self, run_name, override=None, pipeline_path="ml_pipeline/"):
        """Base pipeline class for training and inference.

        Parameters
        ----------
        run_name : str
            Name of this run.
        override : dict, optional
            Hydra config parameters, by default None.
        pipeline_path : str, optional
            Path for saved pipeline, by default "ml_pipeline/".
        """
        self.run_name = run_name
        self.override = override
        self.pipeline_path = pipeline_path
        self.pipeline = dict()
        self.fitted, self.infered = False, False

    def build_train_pipeline(self):
        raise NotImplemented

    def build_inference_pipeline(self):
        raise NotImplemented

    def fit(self):
        raise NotImplemented

    def infer(self):
        raise NotImplemented

    def __str__(self):
        return self.pipeline
