# basic imports
from abc import ABC, abstractmethod
from typing import List
from copy import deepcopy
# mathematical imports
import numpy as np
from scipy import stats
# our imports


class ScenarioGenerator(ABC):

    @abstractmethod
    def seed(self, seed: int) -> None:
        """Sets the random seed for the arrival process. """
        raise NotImplementedError

    @abstractmethod
    def reset(self) -> None:
        """Resets the arrival process"""
        raise NotImplementedError

    @abstractmethod
    def sample_item(self) -> List:
        """Sample new item. """
        raise NotImplementedError


class ItemGenerator(ScenarioGenerator):
    """
    this class creates an item generator
    """

    def sample_item(self):
        return [np.random.rand(), np.random.rand()]

    def seed(self, seed: int) -> None:
        pass

    def reset(self) -> List:
        return self.sample_item()

