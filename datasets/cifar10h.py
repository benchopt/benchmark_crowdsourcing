from benchopt import BaseDataset, safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np
    from pathlib import Path
    import zipfile
    import pooch
    import pandas as pd


class Dataset(BaseDataset):

    name = "cifar10h"
    requirements = ["pip:json", "numpy", "pip:zipfile", "pip:pooch", "pip:pandas"]
    install_cmd = "conda"

    def prepare_data(self):
        self.DIR = Path(__file__).parent.resolve()
        url = (
            "https://github.com/jcpeterson/cifar-10h/"
            "blob/master/data/cifar10h-raw.zip?raw=true"
        )
        filename = self.DIR / "downloads" / "cifar10h-raw.zip"
        filename.parent.mkdir(exist_ok=True)
        if not filename.exists():
            pooch.retrieve(url=url, known_hash=None, fname=filename)
        with zipfile.ZipFile(filename, "r") as zip_ref:
            zip_ref.extractall(self.DIR / "downloads")

    def get_crowd_labels(self):
        csvfile = "cifar10h-raw.csv"
        df = pd.read_csv(self.DIR / "downloads" / csvfile, na_values="-99999")
        df = df[df.is_attn_check == 0]
        uni = df.cifar10_test_test_idx.unique()
        res = {}
        gt = []
        for t in sorted(uni):
            tmp = df[df.cifar10_test_test_idx == t]
            res[int(t)] = {}
            gt.append(tmp.true_label.iloc[0])
            for w in tmp.annotator_id:
                res[int(t)][str(w)] = int(
                    tmp[tmp.annotator_id == w].chosen_label.iloc[0]
                )
        self.gt = np.array(gt).astype(int)
        self.answers = res

    def get_data(self):
        self.DIR = Path(__file__).parent.resolve()
        self.DIRdata = self.DIR / "downloads" / "cifar10h-raw.csv"
        if not (self.DIRdata / "answers.txt").exists():
            self.prepare_data()
        self.get_crowd_labels()
        return dict(
            votes=self.answers,
            ground_truth=self.gt,
            n_worker=2571,
            n_task=10000,
            n_classes=10,
        )
