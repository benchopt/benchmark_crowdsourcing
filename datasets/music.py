from benchopt import BaseDataset, safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np
    from pathlib import Path
    from urllib import request
    import tarfile
    import pandas as pd


class Dataset(BaseDataset):

    name = "Music"
    requirements = ["pip:json", "numpy", "pip:tarfile", "pip:pandas"]
    install_cmd = "conda"

    def prepare_data(self):
        filename = self.DIR / "downloads" / "mturk" / "mturk-datasets.tar.gz"
        filename.parent.mkdir(exist_ok=True)
        target_path = self.DIR / "data"
        target_path.mkdir(exist_ok=True)
        if not filename.exists():
            with request.urlopen(
                request.Request(
                    "http://fprodrigues.com/mturk-datasets.tar.gz",
                    headers={  # not a bot
                        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64)"
                        "AppleWebKit/537.36 (KHTML, like Gecko)"
                        "Chrome/51.0.2704.103 Safari/537.36",
                        "Accept": "text/html,application/xhtml+xml,"
                        "application/xml;q=0.9,*/*;q=0.8",
                    },
                ),
                timeout=60.0,
            ) as response:
                if response.status == 200:
                    with open(filename, "wb") as f:
                        f.write(response.read())
            with tarfile.open(filename, "r:gz") as tar:
                tar.extractall(path=target_path)

    def get_crowd_labels(self):
        self.DIRturk = self.DIR / "data" / "music_genre_classification"
        self.mturk_answers = pd.read_csv(self.DIRturk / "mturk_answers.csv")
        gold = pd.read_csv(self.DIRturk / "music_genre_gold.csv")
        gold = gold[["id", "class"]]

        self.class_to_idx = {
            "blues": 0,
            "classical": 1,
            "country": 2,
            "disco": 3,
            "hiphop": 4,
            "jazz": 5,
            "metal": 6,
            "pop": 7,
            "reggae": 8,
            "rock": 9,
        }
        res = {}
        workers = self.mturk_answers.WorkerID.unique()
        worker_conv = {k: v for k, v in zip(workers, range(len(workers)))}
        tasks = self.mturk_answers["Input.song"].unique()
        tasks_conv = {k: v for k, v in zip(tasks, range(len(tasks)))}

        gt = []
        for _, task in self.mturk_answers.iterrows():
            worker = task["WorkerID"]
            name = task["Input.song"]
            task_id = tasks_conv[name]
            lab = self.class_to_idx[task["Answer.pred_label"]]
            if not res.get(task_id, None):
                res[task_id] = {}
                gt.append(
                    self.class_to_idx[
                            gold[gold["id"] == name]["class"].iloc[0]
                        ]
                    )
            res[task_id][worker_conv[worker]] = lab
        self.answers = res
        self.ground_truth = np.array(gt).astype(int)

    def get_data(self):
        self.DIR = Path(__file__).parent.resolve()
        self.DIRdata = self.DIR / "data" / "music_genre_classification"
        if not (self.DIRdata / "mturk_answers.csv").exists():
            self.prepare_data()
        self.get_crowd_labels()
        return dict(
            votes=self.answers,
            ground_truth=self.ground_truth,
            n_worker=44,
            n_task=700,
            n_classes=10,
        )
