from gym import Env
from src.utils.random_utils import process_random


class SingleMachineWithDeadline(Env):

    def __init__(self,
                 maximal_num_jobs,
                 rnd=0,
                 method=""):
        self.random = process_random(rnd)
        self.num_jobs= self.random.randint(maximal_num_jobs)
        self.release = []
        self.lengths = []
        self.due_time = []
        for _ in range(self.num_jobs):
            self.release.append(self.random.uniform())
            self.lengths.append(self.random.uniform())
            self.due_time.append(self.random.uniform())
