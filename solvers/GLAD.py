from benchopt import BaseSolver, safe_import_context
from benchopt.stopping_criterion import SufficientProgressCriterion

with safe_import_context() as import_ctx:
    import numpy as np
    import scipy as sp
    import scipy.stats
    import scipy.optimize


class Solver(BaseSolver):
    """GLAD algorithm."""

    # Adapted from https://github.com/notani/python-glad
    name = "GLAD"
    install_cmd = "conda"
    requirements = ["numpy", "scipy"]
    stopping_strategy = "callback"
    stopping_criterion = SufficientProgressCriterion(
        patience=10, strategy="callback"
    )

    @staticmethod
    def sigmoid(x):
        return np.piecewise(
            x,
            [x > 0],
            [
                lambda i: 1 / (1 + np.exp(-i)),
                lambda i: np.exp(i) / (1 + np.exp(i)),
            ],
        )

    @staticmethod
    def logsigmoid(x):
        return np.piecewise(
            x,
            [x > 0],
            [
                lambda i: -np.log(1 + np.exp(-i)),
                lambda i: i - np.log(1 + np.exp(i)),
            ],
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

    def run(self, callback):
        self.labels = np.zeros((self.n_task, self.n_workers))
        for task, ans in self.answers.items():
            for worker, lab in ans.items():
                self.labels[task, worker] = lab + 1

        # Initialize priors
        self.priorZ = np.array([1 / self.n_classes] * self.n_classes)
        self.priorAlpha = np.ones(self.n_workers)
        self.priorBeta = np.ones(self.n_task)
        self.probZ = np.empty((self.n_task, self.n_classes))
        self.beta = np.empty(self.n_task)
        self.alpha = np.empty(self.n_workers)

        # Initialize parameters to starting values
        self.alpha = self.priorAlpha.copy()
        self.beta = self.priorBeta.copy()
        self.probZ[:] = self.priorZ[:]

        counter = 0
        while callback(self.probZ):
            self.EStep()
            self.MStep()
            counter += 1
        self.k = counter

    def calcLogProbL(self, item, *args):
        j = int(item[0])
        delta = args[0][j]
        noResp = args[1][j]
        oneMinusDelta = (~delta) & (~noResp)
        exponents = item[1:]
        correct = self.logsigmoid(exponents[delta]).sum()
        wrong = (
            self.logsigmoid(-exponents[oneMinusDelta])
            - np.log(float(self.n_classes - 1))
        ).sum()
        return correct + wrong

    def EStep(self):
        ab = np.array([np.exp(self.beta)]).T @ np.array([self.alpha])
        ab = np.c_[np.arange(self.n_task), ab]

        for k in range(self.n_classes):
            self.probZ[:, k] = np.apply_along_axis(
                self.calcLogProbL,
                1,
                ab,
                (self.labels == k + 1),
                (self.labels == 0),
            )

        # Exponentiate and renormalize
        self.probZ = np.exp(self.probZ)
        s = self.probZ.sum(axis=1)
        self.probZ = (self.probZ.T / s).T

    def packX(self):
        return np.r_[self.alpha.copy(), self.beta.copy()]

    def unpackX(self, x):
        self.alpha = x[: self.n_workers].copy()
        self.beta = x[self.n_workers :].copy()

    def getBoundsX(self, alpha=(-100, 100), beta=(-100, 100)):
        alpha_bounds = np.array(
            [[alpha[0], alpha[1]] for i in range(self.n_workers)]
        )
        beta_bounds = np.array(
            [[beta[0], beta[1]] for i in range(self.n_workers)]
        )
        return np.r_[alpha_bounds, beta_bounds]

    def f(self, x):
        """Return the value of the objective function"""
        self.unpackX(x)
        return -self.computeQ()

    def df(self, x):
        """Return gradient vector"""
        self.unpackX(x)
        dQdAlpha, dQdBeta = self.gradientQ()
        return np.r_[-dQdAlpha, -dQdBeta]

    def MStep(self):
        initial_params = self.packX()
        params = sp.optimize.minimize(
            fun=self.f,
            x0=initial_params,
            method="CG",
            jac=self.df,
            tol=0.01,
            options={"maxiter": 25},
        )
        self.unpackX(params.x)

    def computeQ(self):
        """Calculate the expectation of the joint likelihood"""
        Q = 0
        Q += (self.probZ * np.log(self.priorZ)).sum()
        ab = np.array([np.exp(self.beta)]).T @ np.array([self.alpha])
        logSigma = self.logsigmoid(ab)
        idxna = np.isnan(logSigma)
        if np.any(idxna):
            logSigma[idxna] = ab[idxna]
        logOneMinusSigma = self.logsigmoid(-ab) - np.log(
            float(self.n_classes - 1)
        )
        idxna = np.isnan(logOneMinusSigma)
        if np.any(idxna):
            logOneMinusSigma[idxna] = -ab[idxna]

        for k in range(self.n_classes):
            delta = self.labels == k + 1
            Q += (self.probZ[:, k] * logSigma.T).T[delta].sum()
            oneMinusDelta = (self.labels != k + 1) & (self.labels != 0)
            Q += (self.probZ[:, k] * logOneMinusSigma.T).T[oneMinusDelta].sum()
        Q += np.log(sp.stats.norm.pdf(self.alpha - self.priorAlpha)).sum()
        Q += np.log(sp.stats.norm.pdf(self.beta - self.priorBeta)).sum()
        if np.isnan(Q):
            return -np.inf
        return Q

    def dAlpha(self, item, *args):
        i = int(item[0])
        sigma_ab = item[1:]
        delta = args[0][:, i]
        noResp = args[1][:, i]
        oneMinusDelta = (~delta) & (~noResp)

        probZ = args[2]

        correct = (
            probZ[delta] * np.exp(self.beta[delta]) * (1 - sigma_ab[delta])
        )
        wrong = (
            probZ[oneMinusDelta]
            * np.exp(self.beta[oneMinusDelta])
            * (-sigma_ab[oneMinusDelta])
        )
        # Note: The derivative in Whitehill et al.'s appendix
        # has the term ln(K-1), which is incorrect.

        return correct.sum() + wrong.sum()

    def dBeta(self, item, *args):
        j = int(item[0])
        sigma_ab = item[1:]
        delta = args[0][j]
        noResp = args[1][j]
        oneMinusDelta = (~delta) & (~noResp)

        probZ = args[2][j]

        correct = probZ * self.alpha[delta] * (1 - sigma_ab[delta])
        wrong = probZ * self.alpha[oneMinusDelta] * (-sigma_ab[oneMinusDelta])

        return correct.sum() + wrong.sum()

    def gradientQ(self):

        dQdAlpha = -(self.alpha - self.priorAlpha)
        dQdBeta = -(self.beta - self.priorBeta)

        ab = np.array([np.exp(self.beta)]).T @ np.array([self.alpha])

        sigma = self.sigmoid(ab)
        sigma[np.isnan(sigma)] = 0

        labelersIdx = np.arange(self.n_workers).reshape((1, self.n_workers))
        sigma = np.r_[labelersIdx, sigma]
        sigma = np.c_[np.arange(-1, self.n_task), sigma]

        for k in range(self.n_classes):
            dQdAlpha += np.apply_along_axis(
                self.dAlpha,
                0,
                sigma[:, 1:],
                (self.labels == k + 1),
                (self.labels == 0),
                self.probZ[:, k],
            )

            dQdBeta += (
                np.apply_along_axis(
                    self.dBeta,
                    1,
                    sigma[1:],
                    (self.labels == k + 1),
                    (self.labels == 0),
                    self.probZ[:, k],
                )
                * np.exp(self.beta)
            )

        return dQdAlpha, dQdBeta

    def get_result(self):
        return self.T

    @staticmethod
    def get_next(stop_val):
        return stop_val + 1
