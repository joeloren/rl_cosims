import copy

import numpy as np
from matplotlib import pyplot as plt
from scipy import stats
from scipy.stats import rv_continuous


class MixtureModel(rv_continuous):
    """
    this class creates a mixture of cvrp_distributions using the scipy random generator library.
    """

    def __init__(self, submodels: list, weights: list, a: int, b: int, *args, **kwargs):
        """
        param: submodels - list of submodels each one a scipy random generator
        param: weights - list of weights for each submodel distribution (how often each
        distribution is chosen)
        """
        super().__init__(*args, **kwargs)
        self.submodels = submodels
        self.a = a
        self.b = b
        # make sure weights always adds up to 1 exactly -
        self.weights = list(np.array(weights) / np.sum(np.array(weights)))
        # add all init variables to _ctor_param , this is used for __deepcopy__ method in scipy
        # for more info see line 618 in _distn_infrastructure.py -
        # https://github.com/scipy/scipy/blob/bf4e01b5862a8f20dd79f799ac2330f40cb93897/scipy
        # /stats/_distn_infrastructure.py#L624
        self._ctor_param["submodels"] = self.submodels
        self._ctor_param["weights"] = self.weights

    def _pdf(self, x, *args):
        """
        this function calculates the pdf based on the input x (np.ndarray)
        this is used by scipy to calculate all other functionalities of the distribtuion
        """
        return sum([submodel.pdf(x) * weight for submodel, weight in zip(self.submodels, self.weights)])

    def rvs(self, size):
        """
        this function chooses random nubmers from the distribution, size is the size of array
        returned
        param: size - number of random numbers to generate
        """
        n_components = len(self.weights)
        submodel_choices = np.random.choice(n_components, size=size, p=self.weights)
        submodel_samples = [submodel.rvs(size=size) for submodel in self.submodels]
        rvs = np.choose(submodel_choices, submodel_samples)
        return rvs


class TruncatedGaussian2D(rv_continuous):
    """
    this class creates a truncated 2d gaussian distribution used for positions or any other 2d
    variable in the cvrp_simulation
    """

    def __init__(self, submodels: list, a: int, b: int, *args, **kwargs):
        """
        :param submodels - list of submodels each one a scipy random generator. there are 2
        submodels in this case
        """
        super().__init__(*args, **kwargs)
        self.submodels = submodels
        self.a = a
        self.b = b
        # add all init variables to _ctor_param , this is used for __deepcopy__ method in scipy
        # for more info see line 618 in _distn_infrastructure.py -
        # https://github.com/scipy/scipy/blob/bf4e01b5862a8f20dd79f799ac2330f40cb93897/scipy
        # /stats/_distn_infrastructure.py#L624
        self._ctor_param["submodels"] = self.submodels

    def _pdf(self, x: np.ndarray, *args):
        """
        this will return the pdf of the multi variant normal distribution which is close to the
        pdf of the truncated distrubtion that we want
        """
        pdf = np.prod([submodel.pdf(x) for submodel in self.submodels], axis=0)
        return pdf

    def rvs(self, size: int):
        """
        this function calculates n random numbers based on distribution
        return: np.ndarray of size [Size] or random numbers
        :param size - number of random numbers to generate
        """
        submodel_samples = [submodel.rvs(size=size) for submodel in self.submodels]
        rvs = np.array(submodel_samples)
        return rvs


def main():
    # example for creating a 2d truncated gaussian mixture distribution
    # mu and sigma wanted for the x axis mixture distribution
    norm_params_x_n = np.array([[0.2, 0.02], [0.4, 0.02]])
    # mu and sigma wanted for the y axis mixture distribution
    norm_params_y_n = np.array([[0.6, 0.02], [0.8, 0.03]])
    submodels = []
    for i in range(len(norm_params_x_n)):
        gauss_submodel = []
        a = (0 - norm_params_x_n[i][0]) / norm_params_x_n[i][1]
        b = (1 - norm_params_x_n[i][0]) / norm_params_x_n[i][1]
        gauss_submodel.append(
            stats.truncnorm(a, b, norm_params_x_n[i][0], norm_params_x_n[i][1])
        )
        c = (0 - norm_params_y_n[i][0]) / norm_params_y_n[i][1]
        d = (1 - norm_params_y_n[i][0]) / norm_params_y_n[i][1]
        gauss_submodel.append(
            stats.truncnorm(c, d, norm_params_y_n[i][0], norm_params_y_n[i][1])
        )
        truncated_gauss_2d = TruncatedGaussian2D(gauss_submodel, 0, 1)
        submodels.append(truncated_gauss_2d)
    mixture_gauss_2d = MixtureModel(submodels, [0.5, 0.5], 0, 1)
    mix_2 = copy.deepcopy(mixture_gauss_2d)
    mixture_rvs = mixture_gauss_2d.rvs(100)
    x = np.arange(0, 1, 0.001)
    mixture_pdf = mixture_gauss_2d.pdf(x)
    print(mixture_pdf)
    plt.figure()
    plt.plot(x, mixture_pdf)
    plt.figure()
    plt.scatter(mixture_rvs[0, :], mixture_rvs[1, :])
    plt.grid()
    plt.show()

    ###################################################################
    ###################################################################
    # example if you want only 1d truncated gaussian

    # norm_params_n = np.array([[0.2, 0.02],
    #                           [0.4, 0.02],
    #                           [0.6, 0.02],
    #                           [0.8, 0.03]])
    #
    # submodels = []
    # for i in range(len(norm_params_n)):
    #     a = (0 - norm_params_n[i][0]) / norm_params_n[i][1]
    #     b = (1 - norm_params_n[i][0]) / norm_params_n[i][1]
    #     submodels.append(stats.truncnorm(a, b, norm_params_n[i][0], norm_params_n[i][1]))
    # mixture_gaussian_model = MixtureModel(submodels, [0.25, 0.25, 0.25, 0.25])
    # x_axis = np.arange(-1, 1.2, 0.001)
    # mixture_pdf = mixture_gaussian_model.pdf(x_axis)
    # mixture_rvs = mixture_gaussian_model.rvs(1000)
    # plt.scatter(range(mixture_rvs.size), mixture_rvs)
    # plt.grid()
    # plt.figure()
    # plt.plot(x_axis, mixture_pdf)
    # plt.grid()
    # plt.show()


if __name__ == "__main__":
    main()
    print("done!")
