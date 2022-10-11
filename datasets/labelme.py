from benchopt import BaseDataset
from benchopt import safe_import_context


with safe_import_context() as import_ctx:
    import pooch
    import numpy as np
    from download import download


class Dataset(BaseDataset):

    name = "labelme"
    install_cmd = "conda"
    requirements = ["pip:pooch", "numpy", "pip:download", "pip:tqdm"]
    classification_type = "image"

    def __init__(self):
        self.X = None

    def prepare_data(self):
        download(
            "http://fprodrigues.com/deep_LabelMe.tar.gz",
            pooch.os_cache(f"./data/{self.name}"),
            replace=False,
            kind="tar.gz",
        )
        base_dir = pooch.os_cache(f"./data/{self.name}") / "LabelMe"

        self.train_folder = base_dir / "train"
        self.test_folder = base_dir / "test"
        self.val_folder = base_dir / "val"
        labels = base_dir / "answers.txt"
        labels = np.loadtxt(labels)
        y_train_truth = np.loadtxt(base_dir / "labels_train.txt").astype(int)

        votes = {task: {} for task in range(labels.shape[0])}
        for id_, task in enumerate(labels):
            where = np.where(task != -1)[0]
            for worker in where:
                votes[id_][worker] = int(task[worker])
        self.votes = votes
        self.y_train_truth = y_train_truth
        self.n_workers = labels.shape[1]

    def get_data(self):
        self.prepare_data()
        data = dict(
            train=self.train_folder,
            val=self.val_folder,
            test=self.test_folder,
            votes=self.votes,
            y_train_truth=self.y_train_truth,
            n_workers=self.n_workers,
            n_classes=8,
        )
        return data
