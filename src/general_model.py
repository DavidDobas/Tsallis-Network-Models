class Model:
    def __init__(self):
        pass

    def proposal_fn(self):
        raise NotImplementedError

    def prob_ratio(self):
        raise NotImplementedError
    
    def transform_fn(self):
        raise NotImplementedError