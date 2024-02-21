from benchopt import BaseDataset, safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np
    import pandas as pd


class Dataset(BaseDataset):

    name = "WeatherSentiment"
    requirements = ["numpy", "pip:tarfile", "pip:pandas"]
    install_cmd = "conda"

    def prepare_data(self):
        self.data = pd.read_csv(
            "https://eprints.soton.ac.uk/376543/1/" + "WeatherSentiment_amt.csv",
            header=None,
        )
        self.data.columns = ["worker", "task", "label", "gold", "time"]

    def get_crowd_labels(self):
        res = {}
        workers = self.data.worker.unique()
        worker_conv = {k: v for k, v in zip(workers, range(len(workers)))}
        tasks = self.data.task.unique()
        tasks_conv = {k: v for k, v in zip(tasks, range(len(tasks)))}

        gt = []
        for _, task in self.data.iterrows():
            worker = task["worker"]
            name = task["task"]
            task_id = tasks_conv[name]
            lab = task["label"]
            if not res.get(task_id, None):
                res[task_id] = {}
                gt.append(task["gold"])
            res[task_id][worker_conv[worker]] = lab
        self.answers = res
        self.ground_truth = np.array(gt).astype(int)

    def get_data(self):
        self.prepare_data()
        self.get_crowd_labels()
        return dict(
            votes=self.answers,
            ground_truth=self.ground_truth,
            n_worker=110,
            n_task=300,
            n_classes=5,
        )
