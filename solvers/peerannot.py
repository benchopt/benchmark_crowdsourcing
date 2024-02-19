from benchopt import BaseSolver, safe_import_context
from benchopt.utils import profile

with safe_import_context() as import_ctx:
    import os

    # see https://github.com/pytorch/pytorch/issues/78490
    os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
    from peerannot.models import agg_strategies
    from pathlib import Path


class Solver(BaseSolver):
    name = "peerannot"  # majority vote
    install_cmd = "conda"
    requirements = ["pip:peerannot", "pathlib"]
    sampling_strategy = "iteration"
    parameters = {
        "strategy": ["MV", "DS", "GLAD", "WDS", "Wawa", "TwoThird", "PlantNet"],
    }

    def skip(
        self,
        votes,
        n_worker,
        n_task,
        n_classes,
    ):
        return False, None

    def set_objective(
        self,
        votes,
        n_worker,
        n_task,
        n_classes,
    ):
        folder = Path.cwd() / "tmp"
        folder.mkdir(parents=True, exist_ok=True)
        self.votes = votes
        self.n_worker = n_worker
        self.n_task = n_task
        self.n_classes = n_classes
        strat = agg_strategies[self.strategy]
        if self.strategy == "PlantNet":
            self.strat = strat(
                self.votes,
                n_workers=self.n_worker,
                n_task=self.n_task,
                n_classes=self.n_classes,
                dataset=folder,
                AI="ignored",
                authors=None,
                scores=None,
                alpha=0.5,
                beta=0.2,
            )
        else:
            self.strat = strat(
                self.votes,
                n_workers=self.n_worker,
                n_task=self.n_task,
                n_classes=self.n_classes,
                dataset=folder,
            )

    @profile
    def run(self, maxiter):
        if hasattr(self.strat, "run"):
            if self.strategy in ["WDS", "WAWA", "MV", "TwoThird"]:
                self.strat.run()
            else:
                self.strat.run(maxiter=maxiter)

        self.y_hat = self.strat.get_answers()

    # Return the solution estimate computed.
    def get_result(self):
        return {"yhat": self.y_hat}
