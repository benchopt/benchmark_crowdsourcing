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
        convert_labels = {0: 2, 1: 3, 2: 7, 3: 6, 4: 1, 5: 0, 6: 4, 7: 5}
        votes = {task: {} for task in range(labels.shape[0])}
        for id_, task in enumerate(labels):
            where = np.where(task != -1)[0]
            for worker in where:
                votes[id_][worker] = convert_labels[int(task[worker])]
        self.votes = votes
        self.reorder_tasks()
        self.y_train_truth = self.targets
        self.n_workers = labels.shape[1]

    def get_data(self):
        self.prepare_data()
        data = dict(
            train=self.samples,  # list of [(imagespath, target)]
            val=self.val_folder,  # string to folder with images
            test=self.test_folder,  # string to folder with images
            votes=self.votes,
            y_train_truth=self.y_train_truth,
            n_workers=self.n_workers,
            n_classes=8,
        )
        return data

    def reorder_tasks(self, split="train"):
        task_files = np.loadtxt(
            pooch.os_cache(f"./data/{self.name}")
            / "LabelMe"
            / f"filenames_{split}.txt",
            dtype=str,
        )
        task_labels = np.loadtxt(
            pooch.os_cache(f"./data/{self.name}")
            / "LabelMe"
            / f"labels_{split}_names.txt",
            dtype=str,
        )
        samples = []
        targets = []
        self.class_to_idx = {
            "coast": 0,
            "forest": 1,
            "highway": 2,
            "insidecity": 3,
            "mountain": 4,
            "opencountry": 5,
            "street": 6,
            "tallbuilding": 7,
        }
        for taskf, targetname in zip(task_files, task_labels):
            target = self.class_to_idx[targetname]
            targets.append(target)
            samples.append(
                (
                    pooch.os_cache(f"./data/{self.name}") / targetname,
                    taskf,
                    target,
                )
            )
        self.samples = samples
        self.imgs = samples
        self.targets = targets
