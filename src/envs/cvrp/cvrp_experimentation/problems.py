import numpy as np
from src.envs.cvrp.cvrp_distributions.mixture_distribution import MixtureModel, TruncatedGaussian2D
from src.envs.cvrp.cvrp_simulation.scenario_generator import FixedSample, SampleDynamicBenchmark
from src.envs.cvrp.cvrp_simulation.simulator import CVRPSimulation
from scipy import stats


def create_fixed_static_problem(
    customer_positions: list,
    depot_position: list,
    initial_vehicle_position: list,
    initial_vehicle_capacity: int,
    vehicle_velocity: int,
    customer_demands: list,
    customer_times: list,
):
    """
    Creates a minimal instance with fixed parameters
    :return:
    """
    customer_positions = np.array(customer_positions)
    depot_position = np.array(depot_position)
    initial_vehicle_position = np.array(initial_vehicle_position)
    customer_demands = np.array(customer_demands)
    customer_times = np.array(customer_times)
    max_customers = customer_times.size
    problem_generator = FixedSample(
        depot_position,
        initial_vehicle_position,
        initial_vehicle_capacity,
        vehicle_velocity,
        customer_positions,
        customer_demands,
        customer_times,
    )
    sim = CVRPSimulation(max_customers=max_customers, problem_generator=problem_generator)
    return sim


def create_uniform_dynamic_problem(
    max_customer_times,
    size: int,
    vehicle_velocity: int,
    vehicle_capacity: int,
    max_demand: int,
    random_seed: int,
    start_at_depot: bool,
):
    np.random.seed(random_seed)
    depot_position_rv = stats.uniform(loc=0, scale=1)
    vehicle_position_rv = stats.uniform(loc=0, scale=1)
    customer_positions_rv = stats.uniform(loc=0, scale=1)
    customer_demands_rv = stats.randint(low=1, high=max_demand)
    if max_customer_times == 0:
        # need to take care of this case separately.
        customer_times_rv = type("customer_times_rv_class", (object,), {"rvs": np.zeros})()
    else:
        customer_times_rv = stats.randint(low=0, high=max_customer_times)
    dynamic_generator = SampleDynamicBenchmark(
        depot_position_rv=depot_position_rv,
        vehicle_position_rv=vehicle_position_rv,
        vehicle_capacity=vehicle_capacity,
        vehicle_velocity=vehicle_velocity,
        customer_positions_rv=customer_positions_rv,
        customer_demands_rv=customer_demands_rv,
        customer_times_rv=customer_times_rv,
        vrp_size=size,
        start_at_depot=start_at_depot,
    )
    sim = CVRPSimulation(max_customers=size, problem_generator=dynamic_generator)
    return sim


def create_mixture_guassian_dynamic_problem(
    max_customer_times: int,
    size: int,
    vehicle_velocity: int,
    vehicle_capacity: int,
    max_demand: int,
    random_seed: int,
    start_at_depot: bool,
    distributions_params: dict,
):
    """
    this function creates a cvrp based on given gaussian parameters for spatial distribution
    :param max_customer_times: maximum time for customer to arrive
    :param size - size of problem (graph size)
    :param vehicle_capacity - initial vehicle capacity (this will also be the capacity when the vehicle returns to depot
    :param vehicle_velocity - vehicle velocity, this determins the cvrp time since when the
    vehicle goes to a customer the time will be : distance*vehicle_velocity
    :param max_demand - maximum demand for customer_demand distribution
    :param random_seed - seed for cvrp_distributions random generator
    :param start_at_depot - boolean if vehicle should start at depot or random location
    :param distributions_params - dictionary of distribution parameters for the customer position
    and time (if dynamic) - params: for each parameter that we want to have mixture gaussian there is a
    list of [ mu, sigma] the list as N parameters for each distribution wanted truncated function
        - weights: the weight of each distribution [N_dists] , all weights should add up to 1
    """
    np.random.seed(random_seed)
    # calculating truncated parameters from the original mu,sigma and creating trunc. norm
    # instance with relevant params
    param_customer_positions = distributions_params["params"]["customer_positions"]
    position_trunc_gaussian_2d = get_truncated_2d_submodels(
        param_customer_positions["x"], param_customer_positions["y"], 0, 1
    )
    # creating random variable for mixture gaussian. using a combined distribution for x and y
    # (same as using a 2d mixture distribution)
    customer_positions_rv = MixtureModel(position_trunc_gaussian_2d,
                                         distributions_params["weights"]["customer_positions"], 0, 1,)
    depot_position_rv = stats.uniform(loc=0, scale=1)
    vehicle_position_rv = stats.uniform(loc=0, scale=1)
    customer_demands_rv = stats.randint(low=1, high=max_demand)
    if max_customer_times == 0:
        # need to take care of this case separately.
        customer_times_rv = type("customer_times_rv_class", (object,), {"rvs": np.zeros})()
    else:
        # in this case the dynamic times are taken
        if "customer_times" in distributions_params["weights"].keys():
            # in this case the times should be a mixture gaussian distribution with K gaussians
            param_customer_times = distributions_params["params"]["customer_times"]
            times_trunc_gaussian = get_truncated_submodels(param_customer_times, 0, max_customer_times)
            customer_times_rv = MixtureModel(times_trunc_gaussian, distributions_params["weights"]["customer_times"],
                                             0, max_customer_times)
        else:
            customer_times_rv = stats.randint(low=0, high=max_customer_times)
    dynamic_generator = SampleDynamicBenchmark(
        depot_position_rv=depot_position_rv,
        vehicle_position_rv=vehicle_position_rv,
        vehicle_capacity=vehicle_capacity,
        vehicle_velocity=vehicle_velocity,
        customer_positions_rv=customer_positions_rv,
        customer_demands_rv=customer_demands_rv,
        customer_times_rv=customer_times_rv,
        vrp_size=size,
        start_at_depot=start_at_depot,
    )
    sim = CVRPSimulation(max_customers=size, problem_generator=dynamic_generator)
    return sim


def get_truncated_submodels(params, min_lim, max_lim):
    """
    this function returns a list of truncated normal cvrp_distributions to use in the mixture gaussian
    distribution
    """
    submodels = []
    for i in range(len(params)):
        a = (min_lim - params[i][0]) / params[i][1]
        b = (max_lim - params[i][0]) / params[i][1]
        submodels.append(stats.truncnorm(a, b, params[i][0], params[i][1]))
    return submodels


def get_truncated_2d_submodels(params_x, params_y, min_lim, max_lim):
    """
    this function creates a truncated 2d random variable where the output has 2 dimensions [x, y]
    the number of truncated random variables created is the same as the number of [mu, sigma] in
    params_x and params_y
    :param params_x [N, 2] this is the mu and sigma for the number of cvrp_distributions wanted for
    the mixture gaussian of x
    :param params_y [N, 2] this is the mu and sigma for the number of cvrp_distributions wanted for
    the mixture gaussian of y
    : min_lim [float] the lower limit of the truncated normal distribution (a in scipy language)
    : max_lim [float] the upper limit of the truncated normal distribution (b in scipy language)
    param_x and param_y must have the same length!
    return [N] 2d gaussian cvrp_distributions where each one has a [mu_x, mu_y] and [sigma_x, sigma_y]
    """
    submodels = []
    for i in range(len(params_x)):
        gauss_submodel = []
        a = (min_lim - params_x[i][0]) / params_x[i][1]
        b = (max_lim - params_x[i][0]) / params_x[i][1]
        gauss_submodel.append(stats.truncnorm(a, b, params_x[i][0], params_x[i][1]))
        c = (min_lim - params_y[i][0]) / params_y[i][1]
        d = (max_lim - params_y[i][0]) / params_y[i][1]
        gauss_submodel.append(stats.truncnorm(c, d, params_y[i][0], params_y[i][1]))
        truncated_gauss_2d = TruncatedGaussian2D(gauss_submodel, min_lim, max_lim)
        submodels.append(truncated_gauss_2d)
    return submodels
