from benchopt import BaseDataset
from benchopt import safe_import_context


with safe_import_context() as import_ctx:
    import numpy as np


class Dataset(BaseDataset):

    name = "Simulated"
    install_cmd = "conda"
    requirements = ["scikit-learn"]

    # List of parameters to generate the datasets. The benchmark will consider
    # the cross product for each key in the dictionary.
    parameters = {
        "n_samples": [1000],
        "n_workers": [50],
        "ratio_diff": [2],
        "p_random": [0.2],
        "p_bad": [0.1],
    }

    def __init__(self, n_samples, n_workers, ratio_diff, p_random, p_bad):
        # Store the parameters of the dataset
        self.train, self.val, self.test = None, None, None
        self.n_samples, self.n_workers = n_workers, n_samples
        self.ratio_diff, self.p_random, self.p_bad = (
            ratio_diff,
            p_random,
            p_bad,
        )

    def votes(self, ratio_diff=0.5, p_random=0, p_bad=0.25, seed=42):
        labels = self.rng.choice([0, 1], size=self.n_samples)
        p_hard = (1 - p_random) / (ratio_diff + 1)
        difficulty = self.rng.choice(
            ["easy", "hard", "random"],
            p=[ratio_diff * p_hard, p_hard, p_random],
            size=self.n_samples,
        )
        quality = self.rng.choice(
            ["good", "bad"], size=self.n_workers, p=[1 - p_bad, p_bad]
        )
        answers = {}
        for i in range(self.n_samples):
            answers[i] = {}
            n_ans = self.rng.choice(range(1, self.n_workers))
            who = self.rng.choice(
                range(self.n_workers), size=n_ans, replace=False
            )
            for j in range(n_ans):
                if difficulty[i] == "easy":
                    p_switch = 0
                elif difficulty[i] == "hard":
                    if quality[who[j]] == "bad":
                        p_switch = 0.45
                    else:
                        p_switch = 0.25
                else:  # random
                    p_switch = 0.5
                answers[i][who[j]] = self.rng.choice(
                    [labels[i], 1 - labels[i]], p=[1 - p_switch, p_switch]
                )
        self.answers = answers
        self.y_train_truth = labels

    def get_data(self):
        self.rng = np.random.RandomState(42)
        self.votes(
            ratio_diff=self.ratio_diff,
            p_random=self.p_random,
            p_bad=self.p_bad,
            seed=42,
        )
        return dict(
            train=self.train,
            val=self.val,
            test=self.test,
            votes=self.answers,
            y_train_truth=self.y_train_truth,
            n_workers=self.n_workers,
            n_classes=2,
        )
