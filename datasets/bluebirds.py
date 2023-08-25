from benchopt import BaseDataset
from benchopt import safe_import_context


with safe_import_context() as import_ctx:
    import pooch
    import numpy as np
    import yaml


class Dataset(BaseDataset):

    name = "bluebirds"
    install_cmd = "conda"
    requirements = ["pip:pooch", "numpy"]
    classification_type = "image"

    def prepare_data(self):
        """
        BlueBirds dataset:
            - ground truth in gt.yaml
            - votes in labels.yaml formatted as:
                {worker: {task: vote}}
        """
        odie = pooch.create(
            path=pooch.os_cache(f"./data/{self.name}"),
            base_url="https://raw.githubusercontent.com/welinder/cubam/public/demo/bluebirds/",
            registry={
                "gt.yaml": None,
                "labels.yaml": None,
            },
        )
        train_truth = odie.fetch("gt.yaml")
        votes = odie.fetch("labels.yaml")
        with open(train_truth, "r") as f:
            ground_truth = yaml.safe_load(f)
        self.task_converter = {
            taskid: taskrank
            for taskid, taskrank in zip(
                ground_truth.keys(), range(len(ground_truth))
            )
        }
        self.ground_truth = np.array(list(ground_truth.values()))
        with open(votes, "r") as f:
            labels = yaml.safe_load(f)
        self.worker_converter = {
            workerid: workerrank
            for workerid, workerrank in zip(labels.keys(), range(len(labels)))
        }
        votes = {task: {} for task in self.task_converter.values()}
        for worker, values in labels.items():
            for task, label in values.items():
                votes[self.task_converter[task]][
                    self.worker_converter[worker]
                ] = int(label)
        self.votes = votes

    def get_data(self):
        self.prepare_data()
        data = dict(
            votes=self.votes,
            ground_truth=self.ground_truth,
            n_worker=len(self.worker_converter),
            n_task=len(self.votes),
            n_classes=2,
        )
        return data
