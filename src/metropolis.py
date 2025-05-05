import numpy as np
from tqdm import tqdm

class Metropolis:
    def __init__(self, model, initial_state, steps_between_samples=1e6, progress_bar=True):
        self.state = initial_state
        self.prob_ratio = model.prob_ratio
        self.proposal_fn = model.proposal_fn
        self.transform_fn = model.transform_fn
        self.steps_between_samples = steps_between_samples
        self.random_numbers = None
        self.progress_bar = progress_bar
    def step(self, i):
        proposed_state = self.proposal_fn(self.state)
        if self.random_numbers[i] < self.prob_ratio(proposed_state, self.state):
            self.state = proposed_state
        else:
            del proposed_state

    def sample(self, n_steps):
        samples = []
        self.random_numbers = np.random.rand(int(self.steps_between_samples*n_steps))
        for j in tqdm(range(n_steps), disable=not self.progress_bar):
            for i in range(int(self.steps_between_samples)):
                self.step(i + j * int(self.steps_between_samples))
            samples.append(self.transform_fn(self.state).copy())
        return samples