from benchopt import (
    BaseSolver,
    safe_import_context,
)
from benchopt.stopping_criterion import SufficientDescentCriterion

with safe_import_context() as import_ctx:
    import numpy as np


class Solver(BaseSolver):
    """Majority vote."""

    name = "MV"
    install_cmd = "conda"
    requirements = ["numpy"]

    stopping_criterion = SufficientDescentCriterion(
        strategy="tolerance", patience=1
    )

    def set_objective(
        self, train, val, test, votes, y_train_truth, n_classes, n_workers
    ):
        # The arguments of this function are the results of the
        # `to_dict` method of the objective.
        # They are customizable.
        self.train = train
        self.val = val
        self.test = test
        self.answers = votes
        self.n_classes = n_classes
        self.y_train_truth = y_train_truth
        self.n_workers = n_workers
        self.n_task = len(self.answers)

    def compute_baseline(self):
        baseline = np.zeros((len(self.answers), self.n_classes))
        for task_id in list(self.answers.keys()):
            task = self.answers[task_id]
            for vote in list(task.values()):
                baseline[task_id, vote] += 1
        self.baseline = baseline

    def run(self, tol):
        self.compute_baseline()

    def get_result(self):
        return {"yhat": np.argmax(self.baseline, axis=1), "model": None}
