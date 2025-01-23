"""
N-dim Pose Graph Estimation.

Linear problem where we are estimating the position in N-dim space of a number of nodes.
Linear factors connect each node to the M closest nodes in the space.
The linear factors measure the distance between the nodes in each of the N dimensions.
"""

import numpy as np
import argparse
from matplotlib import animation, pyplot as plt
from gbp import gbp_g2o


np.random.seed(0)


parser = argparse.ArgumentParser()
parser.add_argument("--g2o_file", default="data/tinyGrid3D.g2o",
                    help="g2o style file with POSE3 data")

parser.add_argument("--n_iters", type=int, default=50,
                    help="Number of iterations of GBP")
                    
parser.add_argument("--gauss_noise_std", type=int, default=2,
                    help="Standard deviation of Gaussian noise of measurement model.")
parser.add_argument("--loss", default=None,
                    help="Loss function: None (squared error), huber or constant.")
parser.add_argument("--Nstds", type=float, default=3.,
                    help="If loss is not None, number of stds at which point the "
                         "loss transitions to linear or constant.")
parser.add_argument("--beta", type=float, default=0.01,
                    help="Threshold for the change in the mean of adjacent beliefs for "
                         "relinearisation at a factor.")
parser.add_argument("--num_undamped_iters", type=int, default=6,
                    help="Number of undamped iterations at a factor node after relinearisation.")
parser.add_argument("--min_linear_iters", type=int, default=3,
                    help="Minimum number of iterations between consecutive relinearisations of a factor.")
parser.add_argument("--eta_damping", type=float, default=0.4,
                    help="Max damping of information vector of messages.")

parser.add_argument("--prior_std_weaker_factor", type=float, default=50.,
                    help="Ratio of std of information matrix at measurement factors / "
                         "std of information matrix at prior factors.")

parser.add_argument("--float_implementation", action='store_true', default=False,
                    help="Float implementation, so start with strong priors that are weakened")
parser.add_argument("--final_prior_std_weaker_factor", type=float, default=100.,
                    help="Ratio of information at measurement factors / information at prior factors "
                         "after the priors are weakened (for floats implementation).")
parser.add_argument("--num_weakening_steps", type=int, default=5,
                    help="Number of steps over which the priors are weakened (for floats implementation)")



args = parser.parse_args()
print('Configs: \n', args)
configs = dict({
    'gauss_noise_std': args.gauss_noise_std,
    'loss': args.loss,
    'Nstds': args.Nstds,
    'beta': args.beta,
    'num_undamped_iters': args.num_undamped_iters,
    'min_linear_iters': args.min_linear_iters,
    'eta_damping': args.eta_damping,
    'prior_std_weaker_factor': args.prior_std_weaker_factor,
           })

if args.float_implementation:
    configs['final_prior_std_weaker_factor'] = args.final_prior_std_weaker_factor
    configs['num_weakening_steps'] = args.num_weakening_steps
    weakening_factor = np.log10(args.final_prior_std_weaker_factor) / args.num_weakening_steps

graph = gbp_g2o.create_g2o_graph(args.g2o_file, configs)

graph.generate_priors_var(weaker_factor=args.prior_std_weaker_factor)
graph.update_all_beliefs()

fig = plt.figure()
ax = fig.add_subplot(211, projection='3d')
energy_ax = fig.add_subplot(212)

def update(frame):
    ax.clear()

    energy = graph.energy()
    print(f'Iteration {frame} // Energy {energy}')

    for var_node in graph.var_nodes:
        ax.scatter(var_node.belief.mu[0], var_node.belief.mu[1], var_node.belief.mu[2], c='r', marker='o')
        ax.text(var_node.belief.mu[0], var_node.belief.mu[1], var_node.belief.mu[2], 
                f'{var_node.variableID}', fontsize=12)
    
    graph.synchronous_iteration(robustify=True, local_relin=True)

    graph.save_to_g2o('data/optimized.g2o')

ani = animation.FuncAnimation(fig, update, frames=args.n_iters, repeat=False)

# save to g2o

# Save the animation as a video file
# ani.save('posegraph_contraction.mp4', writer='ffmpeg', fps=1)

plt.show()