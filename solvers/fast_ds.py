# git+https://github.com/sukrutrao/Fast-Dawid-Skene@master

from benchopt import BaseSolver, safe_import_context
from benchopt.utils import profile

with safe_import_context() as import_ctx:
    import numpy as np


class Solver(BaseSolver):
    name = "FDS"  # https://github.com/sukrutrao/Fast-Dawid-Skene/
    install_cmd = "conda"
    sampling_strategy = "iteration"

    references = [
        """Vaibhav B Sinha, Sukrut Rao, Vineeth N Balasubramanian. Fast
        Dawid-Skene: A Fast Vote Aggregation Scheme for Sentiment
        Classification. In Workshop on Issues of Sentiment Discovery and
        Opinion Mining (WISDOM) at the ACM SIGKDD Conference on Knowledge
        Discovery and Data Mining (KDD) 2018, August 2018."""
    ]
    parameters = {
        "strategy": ["FDS", "H"],  # hybrid or fast version
    }

    def set_objective(
        self,
        votes,
        n_worker,
        n_task,
        n_classes,
    ):
        self.votes = votes
        self.n_worker = n_worker
        self.n_task = n_task
        self.n_classes = n_classes
        (
            self.questions,
            self.participants,
            self.classes,
            self.counts,
        ) = self.responses_to_counts(self.votes)

    @profile
    def run(self, maxiter):
        counts = self.counts
        question_classes = self.initialize(counts)
        for _ in range(maxiter):
            (class_marginals, error_rates) = self.m_step(counts, question_classes)
            question_classes = self.e_step(counts, class_marginals, error_rates)
        self.y_hat = np.argmax(question_classes, axis=1)

    def e_step(self, counts, class_marginals, error_rates):
        [self.n_task, self.n_worker, self.n_classes] = np.shape(counts)

        question_classes = np.zeros([self.n_task, self.n_classes])
        final_classes = np.zeros([self.n_task, self.n_classes])

        for i in range(self.n_task):
            for j in range(self.n_classes):
                estimate = class_marginals[j]
                estimate *= np.prod(np.power(error_rates[:, j, :], counts[i, :, :]))

                question_classes[i, j] = estimate
            if self.strategy == "H":
                question_sum = np.sum(question_classes[i, :])
                if question_sum > 0:
                    question_classes[i, :] = question_classes[i, :] / float(
                        question_sum
                    )
            else:
                indices = np.argwhere(
                    question_classes[i, :] == np.max(question_classes[i, :])
                ).flatten()
                final_classes[i, np.random.choice(indices)] = 1

        if self.strategy == "H":
            return question_classes
        else:
            return final_classes

    def m_step(self, counts, question_classes):
        [self.n_task, self.n_worker, self.n_classes] = np.shape(counts)
        class_marginals = np.sum(question_classes, 0) / float(self.n_task)
        error_rates = np.zeros([self.n_worker, self.n_classes, self.n_classes])
        for k in range(self.n_worker):
            for j in range(self.n_classes):
                for ell in range(self.n_classes):
                    error_rates[k, j, ell] = np.dot(
                        question_classes[:, j], counts[:, k, ell]
                    )
                sum_over_responses = np.sum(error_rates[k, j, :])
                if sum_over_responses > 0:
                    error_rates[k, j, :] = error_rates[k, j, :] / float(
                        sum_over_responses
                    )

        return (class_marginals, error_rates)

    def responses_to_counts(self, responses):
        questions = responses.keys()
        questions = sorted(questions)
        participants = set()
        classes = set()
        for i in questions:
            i_participants = responses[i].keys()
            for k in i_participants:
                if k not in participants:
                    participants.add(k)
                ik_responses = [responses[i][k]]
                classes.update(ik_responses)
        classes = list(classes)
        classes.sort()
        participants = list(participants)
        participants.sort()
        counts = np.zeros([self.n_task, self.n_worker, self.n_classes])
        for question in questions:
            i = questions.index(question)
            for participant in responses[question].keys():
                k = participants.index(participant)
                for response in [responses[question][participant]]:
                    j = classes.index(response)
                    counts[i, k, j] += 1
        return (questions, participants, classes, counts)

    def initialize(self, counts):
        [nQuestions, nParticipants, nClasses] = np.shape(counts)
        response_sums = np.sum(counts, 1)
        question_classes = np.zeros([nQuestions, nClasses])
        if self.strategy == "FDS":
            for p in range(nQuestions):
                indices = np.argwhere(
                    response_sums[p, :] == np.max(response_sums[p, :])
                ).flatten()
                question_classes[p, np.random.choice(indices)] = 1
        else:
            for p in range(nQuestions):
                question_classes[p, :] = response_sums[p, :] / np.sum(
                    response_sums[p, :], dtype=float
                )
        return question_classes

    def get_result(self):
        return {"yhat": self.y_hat}
