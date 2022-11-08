from benchopt import (
    BaseSolver,
    safe_import_context,
)

with safe_import_context() as import_ctx:
    import numpy as np


class Solver(BaseSolver):
    """Majority vote."""

    name = "MV"
    install_cmd = "conda"
    requirements = ["numpy"]
    stopping_strategy = "callback"

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
        self.run_aggregation()

    def run_aggregation(self):
        self.compute_baseline()

    def compute_baseline(self):
        baseline = np.zeros((len(self.answers), self.n_classes))
        for task_id in list(self.answers.keys()):
            task = self.answers[task_id]
            for vote in list(task.values()):
                baseline[task_id, vote] += 1
        self.baseline = baseline
        self.y_hat = np.argmax(self.baseline, axis=1)

    def run(self, callback):
        dict_callback = {
            "yhat": self.y_hat,
            "model": None,
        }
        callback(dict_callback)

    def get_result(self):
        return {"yhat": self.y_hat, "model": None}
