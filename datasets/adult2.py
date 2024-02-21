from benchopt import BaseDataset
from benchopt import safe_import_context

with safe_import_context() as import_ctx:
    import pooch
    import numpy as np
    import pandas as pd


class Dataset(BaseDataset):

    name = "AdultContent2"
    install_cmd = "conda"
    requirements = ["pip:pooch", "numpy", "pandas"]
    classification_type = "text"

    def prepare_data(self):
        """
        AdultContent2 dataset:
            - partial ground truth in gold.txt
            - votes in labels.txt formatted as:
                AMTid url classcode
        """
        train_truth = pooch.retrieve(
            "https://raw.githubusercontent.com/ipeirotis/"
            "Get-Another-Label/master/data/AdultContent2/gold.txt",
            known_hash=None,
            path=pooch.os_cache(f"./data/{self.name}"),
        )
        votes = pooch.retrieve(
            "https://raw.githubusercontent.com/ipeirotis/Get-Another-Label/"
            "master/data/AdultContent2/labels.txt",
            known_hash=None,
            path=pooch.os_cache(f"./data/{self.name}"),
        )

        labels = {key: code for (code, key) in enumerate(
            ["G", "P", "R", "X", "B"])}
        train_truth = pd.read_csv(train_truth, header=None, sep="\t")
        df = pd.read_csv(votes, header=None, sep="\t")
        df[2] = df[2].replace(labels)
        # handling tasks and partial ground truth
        n_task = df[1].nunique()
        self.task_converter = {
            taskid: taskrank for taskid, taskrank in zip(df[1].unique(), range(n_task))
        }
        gold_truth = train_truth[
            train_truth.set_index([0]).index.isin(df.set_index([1]).index)
        ].copy()
        gold_truth[0].replace(self.task_converter, inplace=True)
        gold_truth[1].replace(labels, inplace=True)
        self.ground_truth = -np.ones(n_task)
        for _, (i, k) in gold_truth.iterrows():
            self.ground_truth[i] = k
        # handling workers votes
        workers = df[0].unique()
        self.worker_converter = {
            workerid: workerrank
            for workerid, workerrank in zip(workers, range(len(workers)))
        }
        votes = {task: {} for task in self.task_converter.values()}
        for idx, (worker, task, label) in df.iterrows():
            votes[self.task_converter[task]
                  ][self.worker_converter[worker]] = label
        self.votes = votes

    def get_data(self):
        self.prepare_data()
        data = dict(
            votes=self.votes,
            ground_truth=self.ground_truth,
            n_worker=len(self.worker_converter),
            n_task=len(self.votes),
            n_classes=5,
        )
        return data
