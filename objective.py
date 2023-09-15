from benchopt import BaseObjective, safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np


class Objective(BaseObjective):
    """Aggregation crowdsourcing."""

    min_benchopt_version = "1.3"
    name = "crowdsourcing"

    def set_data(self, votes, ground_truth, n_task, n_worker, n_classes):
        self.votes = votes
        self.n_worker = n_worker
        self.ground_truth = ground_truth
        self.n_task = n_task
        self.n_classes = n_classes

    def get_one_result(self):
        return {"yhat": np.zeros(self.n_task)}

    def evaluate_result(self, **kwargs):
        yhat = kwargs["yhat"]
        if yhat.ndim == 2:  # argmax with random tie breaker
            y, x = np.where((yhat.T == yhat.max(1)).T)
            aux = np.random.permutation(len(y))
            xa = np.empty_like(x)
            xa[aux] = x
            yhat = xa[
                np.maximum.reduceat(aux, np.where(np.diff(y, prepend=-1))[0])
            ]
        available = np.where(self.ground_truth != -1)
        self.ground_truth = self.ground_truth[available[0]]
        yhat = np.array(yhat[available[0]]).astype(int)
        # AccTrain recovery accuracy
        accuracy = np.mean(yhat == self.ground_truth)
        # F1: micro and macro
        microf1, macrof1 = self.compute_f1scores(self.ground_truth, yhat)
        return dict(value=accuracy, microf1=microf1, macrof1=macrof1)

    def compute_f1scores(self, y_true, y_pred):
        unique_classes = np.unique(np.concatenate((y_true, y_pred)))
        class_mapping = {cls: idx for idx, cls in enumerate(unique_classes)}
        y_true_onehot = np.zeros((len(y_true), self.n_classes))
        y_pred_onehot = np.zeros((len(y_pred), self.n_classes))
        for cls, idx in class_mapping.items():
            y_true_onehot[np.where(y_true == cls), idx] = 1
            y_pred_onehot[np.where(y_pred == cls), idx] = 1
        micro_tp = np.sum(np.logical_and(y_true_onehot, y_pred_onehot))
        micro_fp = np.sum(
            np.logical_and(np.logical_not(y_true_onehot), y_pred_onehot)
        )
        micro_fn = np.sum(
            np.logical_and(y_true_onehot, np.logical_not(y_pred_onehot))
        )

        micro_precision = (
            micro_tp / (micro_tp + micro_fp)
            if (micro_tp + micro_fp) > 0
            else 0
        )
        micro_recall = (
            micro_tp / (micro_tp + micro_fn)
            if (micro_tp + micro_fn) > 0
            else 0
        )
        micro_f1 = (
            2
            * (micro_precision * micro_recall)
            / (micro_precision + micro_recall)
            if (micro_precision + micro_recall) > 0
            else 0
        )
        macro_f1 = 0
        for idx in range(self.n_classes):
            tp = np.sum(
                np.logical_and(y_true_onehot[:, idx], y_pred_onehot[:, idx])
            )
            fp = np.sum(
                np.logical_and(
                    np.logical_not(y_true_onehot[:, idx]),
                    y_pred_onehot[:, idx],
                )
            )
            fn = np.sum(
                np.logical_and(
                    y_true_onehot[:, idx],
                    np.logical_not(y_pred_onehot[:, idx]),
                )
            )
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = (
                2 * (precision * recall) / (precision + recall)
                if (precision + recall) > 0
                else 0
            )
            macro_f1 += f1
        macro_f1 /= self.n_classes
        return micro_f1, macro_f1

    def get_objective(self):
        return dict(
            votes=self.votes,
            n_worker=self.n_worker,
            n_task=self.n_task,
            n_classes=self.n_classes,
        )
