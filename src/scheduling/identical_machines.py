from gym import Env
from src.utils.random_utils import process_random


class IdenticalMachines(Env):

    def __init__(self,
                 jobs_lengths,
                 num_machines,
                 rnd=0):
        self.random = process_random(rnd)
        self.job_lengths = jobs_lengths

    def reset(self):

