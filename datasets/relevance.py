from benchopt import BaseDataset
from benchopt import safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np
    from crowdkit.datasets import load_dataset
    from sklearn.preprocessing import OrdinalEncoder


class Dataset(BaseDataset):

    name = "relevance"
    install_cmd = "conda"
    requirements = ["numpy", "pip:crowd-kit", "pip:scikit-learn"]
    parameters = {
        "dataset": [
            "relevance-2",
            "relevance-5",
            "nist-trec-relevance",
        ],
    }

    def prepare_data(self):
        """
        Relevance datasets: either relevance 2, 5 or nist-trec
        """
        df, df_gt = load_dataset(self.dataset)
        task_enc = OrdinalEncoder()
        worker_enc = OrdinalEncoder()
        label_enc = OrdinalEncoder()

        df["task"] = task_enc.fit_transform(df[["task"]]).astype(int)
        df["worker"] = worker_enc.fit_transform(df[["worker"]]).astype(int)
        df["label"] = label_enc.fit_transform(df[["label"]]).astype(int)

        df_gt = df_gt.reset_index()
        if len(task_enc.categories_[0]) == len(df_gt):
            df_gt["task"] = task_enc.transform(df_gt[["task"]]).astype(int)
            df_gt.rename(columns={"true_label": "label"}, inplace=True)
            df_gt["label"] = label_enc.transform(df_gt[["label"]]).astype(int)
        else:  # not all ground truth is available
            df_gt["task"] = task_enc.transform(df_gt[["task"]]).astype(int)
            df_gt.rename(columns={"true_label": "label"}, inplace=True)
            df_gt["label"] = label_enc.transform(df_gt[["label"]]).astype(int)
            df_gt.rename(columns={"label": "true_label"}, inplace=True)
        temp = df.merge(df_gt, on=["task"], how="left")
        temp = temp.sort_values(by=["task", "worker"])
        temp["true_label"] = temp["true_label"].fillna(-1).astype(int)

        self.ground_truth = np.array(
            list(temp.groupby("task").max()["true_label"].values)
        )

        votes = {task: {} for task in range(len(self.ground_truth))}
        temp = temp[["task", "worker", "label", "true_label"]]
        for _, (task, worker, label, _) in temp.iterrows():
            votes[task][worker] = label
        self.votes = votes
        self.n_worker = len(worker_enc.categories_[0])
        self.n_classes = len(label_enc.categories_[0])

    def get_data(self):
        self.prepare_data()
        data = dict(
            votes=self.votes,
            ground_truth=self.ground_truth,
            n_worker=self.n_worker,
            n_task=len(self.votes),
            n_classes=self.n_classes,
        )
        return data
