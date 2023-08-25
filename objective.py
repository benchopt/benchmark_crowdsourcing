import numpy as np
from numpy.linalg import norm

from benchopt import BaseObjective


class Objective(BaseObjective):
    """Aggregation crowdsourcing."""

    min_benchopt_version = "1.3"
    name = "crowdsourcing"

    def set_data(self, votes, ground_truth, n_task, n_worker, n_classes):
        self.votes = votes
        self.n_worker = n_worker
        self.ground_truth = ground_truth
        self.n_task = n_task
        self.n_classes = n_classes

    def get_one_solution(self):
        return np.zeros(self.n_task)

    def evaluate_result(self, **kwargs):
        yhat = kwargs["yhat"]
        if yhat.ndim == 2:  # argmax with random tie breaker
            y, x = np.where((yhat.T == yhat.max(1)).T)
            aux = np.random.permutation(len(y))
            xa = np.empty_like(x)
            xa[aux] = x
            yhat = xa[
                np.maximum.reduceat(aux, np.where(np.diff(y, prepend=-1))[0])
            ]

        accuracy = np.mean(yhat == self.ground_truth)
        # XXX compute average entropy too

        return dict(value=accuracy)

    def get_objective(self):
        return dict(
            votes=self.votes,
            n_worker=self.n_worker,
            n_task=self.n_task,
            n_classes=self.n_classes,
        )
