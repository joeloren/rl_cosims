import numpy as np
from gym import Env
from src.utils.random_utils import process_random


class Action:

    def __init__(self, from_machine, job_idx, to_machine):
        self.from_machine = from_machine
        self.job_idx = job_idx
        self.to_machine = to_machine


class IdenticalMachines(Env):

    def __init__(self,
                 jobs_lengths,
                 num_machines,
                 maximal_episode,
                 rnd=0,
                 method=""):
        self.random = process_random(rnd)
        self.jobs_lengths = jobs_lengths
        self.num_machines = num_machines
        self.maximal_episode = maximal_episode
        self.queues = None
        self.t = None

    def reset(self):
        self.t = 0
        self.queues = [list()] * self.num_machines
        for job_length in self.jobs_lengths:
            idx_machine = self.random.randint(self.num_machines)
            self.queues[idx_machine].append(job_length)

    def step(self, action):
        previous_time_span = self._compute_time_span()
        job_length = self.queues[action.from_machine]
        self.queues[action.to_machine].append(job_length)
        time_span = self._compute_time_span()
        reward = time_span - previous_time_span
        done = self.t < self.maximal_episode
        info = None
        self.t += 1
        return self.queues, reward, done, None

    def _compute_time_span(self):
        queues_span = [np.sum(self.queues[i]) for i in range(self.num_machines)]
        return np.amax(queues_span)

    def render(self, mode='human'):
        strs = list()
        strs.append(f"num_machines={self.num_machines}; maximal_episode={self.maximal_episode}")

