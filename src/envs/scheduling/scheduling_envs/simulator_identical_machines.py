import numpy as np
from gym import Env
from src.utils.random_utils import process_random


class Action:

    def __init__(self, from_machine, job_idx, to_machine):
        self.from_machine = from_machine
        self.job_idx = job_idx
        self.to_machine = to_machine

    def __str__(self):
        return f"({self.from_machine},{self.job_idx},{self.to_machine})"

    def __repr__(self):
        return self.__str__()


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
        self.queues = [list() for _ in range(self.num_machines)]
        for job_length in self.jobs_lengths:
            idx_machine = self.random.randint(self.num_machines)
            self.queues[idx_machine].append(job_length)

    def step(self, action):
        info = None
        previous_time_span = self._compute_time_span()
        job_length = self.queues[action.from_machine][action.job_idx]
        del self.queues[action.from_machine][action.job_idx]
        self.queues[action.to_machine].append(job_length)
        time_span = self._compute_time_span()
        reward = time_span - previous_time_span
        done = self.t >= self.maximal_episode
        self.t += 1
        return self.queues, reward, done, info

    def _compute_time_span(self):
        queues_span = [np.sum(self.queues[i]) for i in range(self.num_machines)]
        return np.amax(queues_span)

    def render(self, mode='human'):
        strs = list()
        strs.append(f"num_machines={self.num_machines}; maximal_episode={self.maximal_episode}")
        for queue in self.queues:
            qs = list()
            for job in queue:
                qs.append(f"{job}")
            strs.append(",".join(qs))
        return "\n".join(strs)

    def get_possible_actions(self):
        possible_actions = []
        for from_queue in range(len(self.queues)):
            for idx_job in range(len(self.queues[from_queue])):
                for to_queue in range(len(self.queues)):
                    possible_actions.append(Action(from_queue, idx_job, to_queue))
        return possible_actions

    def get_random_action(self):
        prob = [len(queue) for queue in self.queues]
        prob /= np.sum(prob)
        from_machine = self.random.choice(self.num_machines, p=prob)
        job_idx = self.random.choice(len(self.queues[from_machine]))
        to_machine = self.random.choice(self.num_machines)
        return Action(from_machine=from_machine, job_idx=job_idx, to_machine=to_machine)

    @staticmethod
    def observation(obs):
        # the simulator returns obs without any changes (used for wrappers)
        return obs

