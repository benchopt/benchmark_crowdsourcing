from benchopt import BaseSolver, safe_import_context
from benchopt.utils import profile

with safe_import_context() as import_ctx:
    import crowdkit.aggregation.classification as kitagg
    import pandas as pd


class Solver(BaseSolver):
    name = "crowd-kit"
    install_cmd = "conda"
    requirements = ["pip:crowd-kit", "pandas"]

    parameters = {"strategy": ["DawidSkene", "GLAD", "Wawa", "KOS", "MACE", "MMSR"]}

    def skip(
        self,
        votes,
        n_worker,
        n_task,
        n_classes,
    ):
        if n_classes > 2 and self.strategy == "KOS":
            return (
                True,
                f"{self.name}{self.strategy} only handles binary labels",
            )
        return False, None

    def json_to_dataframe(self, json_data):
        rows = []
        for task, worker_data in json_data.items():
            for worker, label in worker_data.items():
                rows.append({"task": task, "worker": worker, "label": label})
        df = pd.DataFrame(rows)
        return df

    def set_objective(
        self,
        votes,
        n_worker,
        n_task,
        n_classes,
    ):
        self.votes = self.json_to_dataframe(votes)
        self.n_worker = n_worker
        self.n_task = n_task
        self.n_classes = n_classes
        self.strat = getattr(kitagg, self.strategy)

    @profile
    def run(self, maxiter):
        if self.strategy in ["Wawa"]:  # non iterative strategies
            self.aggregation = self.strat()
        else:
            self.aggregation = self.strat(n_iter=maxiter)
        self.y_hat = self.aggregation.fit_predict(self.votes)

    # Return the solution estimate computed.
    def get_result(self):
        return {"yhat": self.y_hat.to_numpy(dtype=int)}
