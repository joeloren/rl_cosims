from spinup.utils.run_utils import setup_logger_kwargs
from spinup.algos.pytorch.ppo.ppo import ppo
from spinup.algos.pytorch.ppo.core import MLPActorCritic
from src.envs.online_knapsack.oks_simulation.problem_generator import ItemGenerator
from src.envs.online_knapsack.oks_simulation.simulator import Simulator
from src.training.torch_utils import get_available_device


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--hid', type=int, default=64)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--cpu', type=int, default=4)
    parser.add_argument('--steps', type=int, default=4000)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--exp_name', type=str, default='ppo')
    args = parser.parse_args()

    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)
    generator = ItemGenerator()
    env_generator = lambda : Simulator(max_steps=10, max_capacity=10, problem_generator=generator)
    ac_kwargs = dict(hidden_sizes=[args.hid]*args.l)
    ppo(env_fn=env_generator, actor_critic=MLPActorCritic, ac_kwargs=ac_kwargs, gamma=args.gamma, seed=args.seed,
        steps_per_epoch=args.steps, epochs=args.epochs, logger_kwargs=logger_kwargs)
