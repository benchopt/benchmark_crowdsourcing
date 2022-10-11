from benchopt import BaseSolver, safe_import_context
from benchopt.stopping_criterion import SufficientProgressCriterion

with safe_import_context() as import_ctx:
    import numpy as np


class Solver(BaseSolver):
    """Dawid and Skene confusion matrices EM algorithm."""

    name = "DawidSkene"
    install_cmd = "conda"
    requirements = ["numpy"]
    stopping_strategy = "callback"
    stopping_criterion = SufficientProgressCriterion(
        patience=10, strategy="callback"
    )

    def set_objective(
        self, train, val, test, votes, y_train_truth, n_classes, n_workers
    ):
        # The arguments of this function are the results of the
        # `to_dict` method of the objective.
        # They are customizable.
        self.train = train
        self.val = val
        self.test = test
        self.answers = votes
        self.n_classes = n_classes
        self.y_train_truth = y_train_truth
        self.n_workers = n_workers
        self.n_task = len(self.answers)

    def get_crowd_matrix(self):
        matrix = np.zeros((self.n_task, self.n_workers, self.n_classes))
        for task, ans in self.answers.items():
            for worker, label in ans.items():
                matrix[task, worker, label] += 1
        self.crowd_matrix = matrix

    def init_T(self):
        T = self.crowd_matrix.sum(axis=1)
        tdim = T.sum(1, keepdims=True)
        self.T = np.where(tdim > 0, T / tdim, 0)

    def m_step(self):
        """Maximizing log likelihood (see eq. 2.3 and 2.4 Dawid and Skene 1979)

        Returns:
            p: (p_j)_j probabilities that instance has true response j if drawn
        at random (class marginals)
            pi: number of times worker k records l when j is correct / number
        of instances seen by worker k where j is correct
        """
        p = self.T.sum(0) / self.n_task
        pi = np.zeros((self.n_workers, self.n_classes, self.n_classes))
        for q in range(self.n_classes):
            pij = self.T[:, q] @ self.crowd_matrix.transpose((1, 0, 2))
            denom = pij.sum(1)
            pi[:, q, :] = pij / np.where(denom <= 0, -1e9, denom).reshape(
                -1, 1
            )
        self.p, self.pi = p, pi

    def e_step(self):
        """Estimate indicator variables (see eq. 2.5 Dawid and Skene 1979)
        Returns:
            T: New estimate for indicator variables (n_task, n_worker)
            denom: value used to compute likelihood easily
        """
        T = np.zeros((self.n_task, self.n_classes))
        for i in range(self.n_task):
            for j in range(self.n_classes):
                num = (
                    np.prod(
                        np.power(self.pi[:, j, :], self.crowd_matrix[i, :, :])
                    )
                    * self.p[j]
                )
                T[i, j] = num
        self.denom_e_step = T.sum(1, keepdims=True)
        T = np.where(self.denom_e_step > 0, T / self.denom_e_step, T)
        self.T = T

    def log_likelihood(self):
        return np.log(np.sum(self.denom_e_step))

    def run(self, callback):
        self.get_crowd_matrix()
        self.init_T()
        k = 0
        while callback(self.T):
            self.m_step()
            self.e_step()
            ll = self.log_likelihood()
            k += 1
        self.loglikelihood_value = ll
        self.counter = k

    def get_result(self):
        return self.T

    @staticmethod
    def get_next(stop_val):
        return stop_val + 1
