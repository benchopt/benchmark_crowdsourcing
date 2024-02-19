from benchopt import BaseDataset, safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np
    import pandas as pd
    import pooch
    import zipfile
    from pathlib import Path


class Dataset(BaseDataset):

    name = "AudioBirds"
    requirements = ["numpy", "pip:zipfile", "pip:pandas", "pip:pooch"]
    install_cmd = "conda"

    def prepare_data(self):
        self.DIR = Path(__file__).parent.resolve()
        url = "https://zenodo.org/records/7030863/files/bird_sound_training_data.zip"
        filename = self.DIR / "downloads" / "bird_sound_training_data.zip"
        filename.parent.mkdir(exist_ok=True)
        if not filename.exists():
            pooch.retrieve(url=url, known_hash=None, fname=filename)
        with zipfile.ZipFile(filename, "r") as zip_ref:
            zip_ref.extractall(self.DIR / "downloads")
        self.filename = (
            self.DIR
            / "downloads"
            / "bird_sound_training_data"
            / "letters"
            / "annotations.tsv"
        )
        self.user_expertise = (
            self.DIR / "downloads" / "bird_sound_training_data" / "users.tsv"
        )

    def get_crowd_labels(self):
        self.data = pd.read_csv(self.filename, sep="\t")
        self.user_data = pd.read_csv(self.user_expertise, sep="\t")
        self.conv_tasks = {v: k for k, v in enumerate(set(self.data["candidate_id"]))}
        self.conv_workers = {v: k for k, v in enumerate(set(self.user_data["user_id"]))}
        is_expert = [
            1 if self.user_data.iloc[i]["birdwatching_activity_level"] == 4 else 0
            for i in range(len(self.user_data))
        ]
        answers = {k: {} for k in self.conv_tasks.values()}
        truth = [-1 for _ in range(len(self.conv_tasks))]
        for index, row in self.data.iterrows():
            answers[self.conv_tasks[row["candidate_id"]]][
                self.conv_workers[row["user_id"]]
            ] = int(row["annotation"])
            if is_expert[self.conv_workers[row["user_id"]]]:
                truth[self.conv_tasks[row["candidate_id"]]] = int(row["annotation"])
        self.ground_truth = np.array(truth).astype(int)
        self.answers = answers

    def get_data(self):
        self.prepare_data()
        self.get_crowd_labels()
        return dict(
            votes=self.answers,
            ground_truth=self.ground_truth,
            n_worker=204,
            n_task=79592,
            n_classes=2,
        )
