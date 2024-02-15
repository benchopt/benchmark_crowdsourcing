from benchopt import BaseDataset, safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np
    import json
    from pathlib import Path
    from urllib import request
    import tarfile


class Dataset(BaseDataset):

    name = "LabelMe"
    requirements = ["pip:json", "numpy", "tarfile"]
    install_cmd = "conda"

    def prepare_data(self):
        filename = self.DIR / "downloads" / "labelme_raw.tar.gz"
        filename.parent.mkdir(exist_ok=True)
        target_path = self.DIR / "data"
        target_path.mkdir(exist_ok=True)
        if not filename.exists():
            with request.urlopen(
                request.Request(
                    "http://fprodrigues.com/deep_LabelMe.tar.gz",
                    headers={  # not a bot
                        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/51.0.2704.103 Safari/537.36",
                        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
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
        conv_label = {
            "highway": 0,
            "insidecity": 1,
            "tallbuilding": 2,
            "street": 3,
            "forest": 4,
            "coast": 5,
            "mountain": 6,
            "opencountry": 7,
        }
        crowdlabels = np.loadtxt(self.DIRdata / "answers.txt")
        orig_name = np.loadtxt(self.DIRdata / "filenames_train.txt", dtype=str)
        res_train = {task: {} for task in range(crowdlabels.shape[0])}
        gt = []
        for id_, task in enumerate(crowdlabels):
            where = np.where(task != -1)[0]
            for worker in where:
                res_train[id_][int(worker)] = int(task[worker])
            gt.append(conv_label[orig_name[id_].split("_")[0]])
        self.ground_truth = np.array(gt)
        self.answers = res_train

    def get_data(self):
        self.DIR = Path(__file__).parent.resolve()
        self.DIRdata = self.DIR / "data" / "LabelMe"
        if not (self.DIRdata / "answers.txt").exists():
            self.prepare_data()
        self.get_crowd_labels()
        return dict(
            votes=self.answers,
            ground_truth=self.ground_truth,
            n_worker=77,
            n_task=1000,
            n_classes=8,
        )
