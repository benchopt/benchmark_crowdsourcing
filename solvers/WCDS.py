from benchopt import BaseSolver, safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np


class Solver(BaseSolver):
    """Dawid and Skene confusion matrices EM algorithm."""

    name = "WCDawidSkene"
    install_cmd = "conda"
    requirements = ["numpy"]
    stopping_strategy = "callback"
    parameters = {
        "L": [5, 10],
        "maxiter": [50],
        "epsilon": [1e-5],
    }

    def set_objective(
        self, train, val, test, votes, y_train_truth, n_classes, n_workers
    ):
        self.train = train
        self.val = val
        self.test = test
        self.answers = votes
        self.n_classes = n_classes
        self.y_train_truth = y_train_truth
        self.n_workers = n_workers
        self.n_task = len(self.answers)
        self.run_aggregation()

    def get_crowd_matrix(self):
        matrix = np.zeros((self.n_task, self.n_workers, self.n_classes))
        for task, ans in self.answers.items():
            for worker, label in ans.items():
                matrix[task, worker, label] += 1
        self.crowd_matrix = matrix

    def initialize_parameter(self, x, K, L, random=True, delta=1e-10):
        n = x.shape[0]
        m = x.shape[1]

        if random:
            theta = np.clip(
                np.einsum(
                    "ki->ik", np.einsum("ijk->ki", x) / np.einsum("ijk->i", x)
                )
                + np.random.normal(scale=0.1, size=[n, K]),
                0.0,
                1.0,
            )
            theta = np.einsum(
                "ki->ik", np.einsum("ik->ki", theta) / np.sum(theta, axis=1)
            )
        else:
            theta = np.einsum(
                "ki->ik", np.einsum("ijk->ki", x) / np.einsum("ijk->i", x)
            )

        x += delta
        pi = np.einsum(
            "mjk->jkm",
            np.einsum("ik,ijm->mjk", theta, x)
            / np.einsum("ik,ijm->jk", theta, x),
        )
        order = np.array([np.linalg.norm(pi[j], ord="nuc") for j in range(m)])
        sigma = np.array(
            sorted(
                np.c_[np.arange(m), order],
                key=lambda pair: pair[1],
                reverse=True,
            )
        )[:, 0].astype(dtype=int)
        J = np.array(
            [sigma[int(m * l / L) : int(m * (l + 1) / L)] for l in range(L)]
        )
        lambda_ = np.array(
            [(np.sum(pi[J[l]], axis=0)) * L / m for l in range(L)]
        )

        phi = np.zeros([m, L])
        for l in range(L):
            for j in J[l]:
                phi[j, l] = 1.0

        rho = np.einsum("ik->k", theta) / n
        tau = np.einsum("jl->l", phi) / m
        return theta, phi, rho, tau, lambda_

    def variational_update(
        self, x, theta, phi, rho, tau, lambda_, delta=1e-10
    ):
        theta_prime = np.exp(
            np.einsum("ijm,jl,lkm->ik", x, phi, np.log(lambda_ + delta))
            + np.log(rho + delta)
        )
        phi_prime = np.exp(
            np.einsum("ijm,ik,lkm->jl", x, theta, np.log(lambda_ + delta))
            + np.log(tau + delta)
        )
        theta = np.einsum(
            "ki->ik", theta_prime.T / np.sum(theta_prime.T, axis=0)
        )
        phi = np.einsum("lj->jl", phi_prime.T / np.sum(phi_prime.T, axis=0))
        return theta, phi

    def hyper_parameter_update(self, x, theta, phi):
        n = x.shape[0]
        m = x.shape[1]

        lambda_prime = np.einsum("ik,jl,ijm->mlk", theta, phi, x)
        lambda_ = np.einsum(
            "mlk->lkm", lambda_prime / np.sum(lambda_prime, axis=0)
        )

        rho = np.einsum("ik->k", theta) / n
        tau = np.einsum("jl->l", phi) / m

        return rho, tau, lambda_

    def elbo(self, x, theta, phi, rho, tau, lambda_, delta=1e-10):
        l = (
            np.einsum(
                "ik,jl,ijm,lkm->", theta, phi, x, np.log(lambda_ + delta)
            )
            + np.einsum("ik,k->", theta, np.log(rho + delta))
            + np.einsum("jl,l->", phi, np.log(tau + delta))
            - np.einsum("ik,ik->", theta, np.log(theta + delta))
            - np.einsum("jl,jl->", phi, np.log(phi + delta))
        )
        if np.isnan(l):
            raise ValueError("ELBO is Nan!")
        return l

    def one_iteration(self, x, K, L, random=False):
        theta, phi, rho, tau, lambda_ = self.initialize_parameter(
            x, K, L, random=random
        )
        l = -1e100
        while True:
            theta, phi = self.variational_update(
                x, theta, phi, rho, tau, lambda_
            )
            rho, tau, lambda_ = self.hyper_parameter_update(x, theta, phi)
            l_ = self.elbo(x, theta, phi, rho, tau, lambda_)
            if np.abs(l_ - l) < 1e-4:
                break
            else:
                l = l_
        return theta, phi, lambda_, rho

    def is_chance_rate(self, theta):
        n = self.n_task
        K = self.n_classes
        sum_ = np.sum(theta, axis=0)
        for k in range(K):
            if int(sum_[k]) == n:
                return True
        return False

    def run_aggregation(self):
        self.get_crowd_matrix()
        x = self.crowd_matrix
        K = self.n_classes
        L = self.L
        c = 1
        self.theta, phi, lambda_, rho = self.one_iteration(
            x, K, L, random=False
        )
        while (c < self.maxiter) and (self.is_chance_rate(self.theta)):
            c += 1
            self.theta, phi, lambda_, rho = self.one_iteration(
                x, K, L, random=True
            )
        g_hat = np.argmax(self.theta, axis=1)
        pi_hat = lambda_[np.argmax(phi, axis=1)]
        self.y_hat = g_hat
        self.pi = pi_hat
        self.rho = rho
        self.c = c

    def run(self, callback):
        dict_callback = {"yhat": self.theta, "model": None}
        callback(dict_callback)

    def get_result(self):
        return {"yhat": self.theta, "model": None}

    @staticmethod
    def get_next(stop_val):
        return stop_val + 1
