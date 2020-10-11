from distutils.core import setup

setup(
    name='rl_cosims',
    version='1.0',
    packages=['src', 'src.envs', 'src.envs.cvrp', 'src.envs.cvrp.cvrp_utils', 'src.envs.cvrp.cvrp_wrappers',
              'src.envs.cvrp.cvrp_baselines', 'src.envs.cvrp.cvrp_simulation', 'src.envs.cvrp.cvrp_distributions',
              'src.envs.cvrp.cvrp_experimentation', 'src.envs.scheduling', 'src.envs.graph_coloring',
              'src.envs.graph_coloring.gc_utils', 'src.envs.graph_coloring.gc_baselines',
              'src.envs.graph_coloring.gc_simulation', 'src.envs.graph_coloring.gc_experimentation', 'src.agents',
              'src.models', 'src.training'],
    url='https://github.com/joeloren/rl_cosims',
    license='',
    author='ROS1TV, ORJ1TV, DID1TV',
    author_email='chana.ross@bosch.com, Joel.Oren@bosch.com, Dotan.DiCastro@il.bosch.com',
    description='co online and offline envs (graph coloring, cvrp, scheduling and knapsack), '
                'ppo implementation for torch and torch-geometric'
)
