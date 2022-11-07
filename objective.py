from benchopt import BaseObjective, safe_import_context

# Protect import to allow manipulating objective without importing library
# Useful for autocompletion and install commands
with safe_import_context() as import_ctx:
    import numpy as np


class Objective(BaseObjective):
    name = "Crowdsourcing"

    def get_one_solution(self):
        # Return one solution. This should be compatible with 'self.compute'.
        return {"yhat": np.zeros(len(self.votes)), "model": None}

    def set_data(
        self,
        train,
        val,
        test,
        votes,
        y_train_truth,
        n_classes,
        n_workers,
        **kwargs
    ):
        # The keyword arguments of this function are the keys of the `data`
        # dict in the `get_data` function of the dataset.
        # They are customizable.
        self.train = train
        self.val = val
        self.test = test
        self.votes = votes
        self.y_train_truth = y_train_truth
        self.n_classes = n_classes
        self.n_workers = n_workers

    def compute(self, res):
        # The arguments of this function are the outputs of the
        # `get_result` method of the solver.
        # They are customizable.
        yhat = res["yhat"]
        results = dict()
        if yhat.ndim > 1:
            top1 = np.argmax(yhat, axis=1)
        else:
            top1 = yhat
        train_accuracy = np.mean(top1 == self.y_train_truth)
        results["Train Accuracy"] = train_accuracy
        results["value"] = train_accuracy
        return results

    def to_dict(self):
        # The output of this function are the keyword arguments
        # for the `set_objective` method of the solver.
        # They are customizable.
        return dict(
            train=self.train,
            val=self.val,
            test=self.test,
            votes=self.votes,
            y_train_truth=self.y_train_truth,
            n_classes=self.n_classes,
            n_workers=self.n_workers,
        )
