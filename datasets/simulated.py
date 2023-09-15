from benchopt import BaseDataset, safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np


class Dataset(BaseDataset):

    name = "Simulated"
    install_cmd = "conda"

    # List of parameters to generate the datasets. The benchmark will consider
    # the cross product for each key in the dictionary.
    parameters = {
        "n_task": [200],
        "n_worker": [50],
        "n_classes": [3],
        "ratio": [0.5],  # probability to generate a spammer
    }

    def votes(self):
        rng = np.random.default_rng(42)
        true_labels = rng.choice(
            self.n_classes, size=self.n_task, replace=True
        )
        votes = {}
        type_worker = rng.choice(
            ["hammer", "spammer"],
            size=self.n_worker,
            p=[1 - self.ratio, self.ratio],
        )
        for i in range(self.n_task):
            votes[i] = {}
            for j in range(self.n_worker):
                if type_worker[j] == "hammer":
                    votes[i][j] = true_labels[i]
                else:
                    votes[i][j] = rng.choice(range(1, self.n_classes))
        self.answers = votes
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
