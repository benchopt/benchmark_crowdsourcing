from benchopt import BaseDataset
from benchopt import safe_import_context


with safe_import_context() as import_ctx:
    import pooch
    import numpy as np
    import pandas as pd
    import os
    import tarfile


class Dataset(BaseDataset):

    name = "rte"
    install_cmd = "conda"
    requirements = ["pip:pooch", "numpy", "pandas"]
    classification_type = "text"

    def prepare_data(self):
        """
        rte dataset:
            - ground truth in gt.yaml
            - votes in labels.yaml formatted as:
                {worker: {task: vote}}
        """
        str_ = "snow2008_mturk_data_with_orig_files_assembled_201904.zip"
        odie = pooch.create(
            path=pooch.os_cache(f"./data/{self.name}"),
            base_url="https://sites.google.com/site/nlpannotations/",
            registry={
                str_: None,
            },
        )
        data = odie.fetch(odie.registry_files[0], processor=pooch.Unzip())
        all_collect = [s for s in data if s.endswith("all_collected_data.tgz")]
        tar = tarfile.open(all_collect[0], "r:gz")
        tar.extractall(path=pooch.os_cache(f"./data/{self.name}"))
        tar.close()
        filenames = [
            [
                data[i]
                for i in range(len(data))
                if data[i].endswith("rte1.tsv")
            ][0],
            os.path.join(
                pooch.os_cache(f"./data/{self.name}"), "rte.standardized.tsv"
            ),
        ]
        # load data
        gt = pd.read_csv(filenames[0], sep="\t")
        labels = pd.read_csv(filenames[1], sep="\t")
        # process ground truth
        ground_truth = {}
        for index, row in gt.iterrows():
            ground_truth[row["id"]] = int(row["value"])
        self.task_converter = {
            taskid: taskrank
            for taskid, taskrank in zip(
                ground_truth.keys(), range(len(ground_truth))
            )
        }
        self.ground_truth = np.array(list(ground_truth.values()))
        # process votes and tasks
        all_workers = labels["!amt_worker_ids"].unique()
        self.worker_converter = {
            workerid: workerrank
            for workerid, workerrank in zip(
                all_workers, range(len(all_workers))
            )
        }
        tasks = {task: [] for task in range(len(self.task_converter))}
        votes = {task: {} for task in range(len(self.task_converter))}
        for index, row in labels.iterrows():
            tt = self.task_converter[row["orig_id"]]
            votes[tt][self.worker_converter[row["!amt_worker_ids"]]] = int(
                row["response"]
            )
            if len(tasks[tt]) == 0:
                text_data = gt[gt.id == tt]
                tasks[tt].append([text_data["hypothesis"], text_data["text"]])
        self.votes = votes
        self.tasks = tasks

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
