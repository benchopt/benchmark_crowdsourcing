import numpy as np

from benchopt import BaseDataset


with safe_import_context() as import_ctx:
    import numpy as np
    from peerannot.helpers.simulations_strategies import simulation_strategies
    from pathlib import Path


class Dataset(BaseDataset):

    name = "Simulated"
    install_cmd = "conda"
    requirements = ["pip:peerannot"]

    # List of parameters to generate the datasets. The benchmark will consider
    # the cross product for each key in the dictionary.
    parameters = {
        "n_task": [200],
        "n_worker": [50],
        "ratio": [0.2],  # proportion of spammer
        "n_classes": [3],
        "strategy": ["hammer-spammer"],
        "feedback": [20],
        "workerload": [50],
    }

    def __init__(
        self,
        n_task,
        n_worker,
        n_classes,
        ratio,
        strategy,
        feedback,
        workerload,
    ):
        # Store the parameters of the dataset
        self.n_task, self.n_worker = n_worker, n_task
        self.strategy = strategy
        self.ratio = ratio
        self.feedback, self.workerload = feedback, workerload

    def votes(self):
        rng = np.random.default_rng(42)
        strat = simulation_strategies[self.strategy.lower()]
        true_labels = rng.choice(
            self.n_classes, size=self.n_task, replace=True
        )
        folder = Path.cwd() / "simulation_data"
        folder.mkdir(parents=True, exist_ok=True)
        answers = strat(
            self.n_worker,
            true_labels,
            self.n_classes,
            rng,
            ratio=self.ratio,
            feedback=self.feedback,
            workerload=self.workerload,
            verbose=False,
            imbalance_votes=False,
            folder=folder,
        )
        self.answers = answers
        self.ground_truth = true_labels

    def get_data(self):
        self.votes()
        return dict(
            votes=self.answers,
            ground_truth=self.ground_truth,
            n_worker=self.n_worker,
            n_task=self.n_task,
            n_classes=self.n_classes,
        )

        rng = np.random.RandomState(self.random_state)
        X = rng.randn(self.n_samples, self.n_features)
        y = rng.randn(self.n_samples)

        # `data` holds the keyword arguments for the `set_data` method of the
        # objective.
        # They are customizable.
        return dict(X=X, y=y)
