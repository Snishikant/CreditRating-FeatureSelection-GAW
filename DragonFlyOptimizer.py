# Import from stdlib
import logging
# Import modules
import random

import numpy as np
# Import from package
from pyswarms.backend import compute_pbest
from pyswarms.backend.topology import Ring
from pyswarms.base import DiscreteSwarmOptimizer
from pyswarms.utils.console_utils import cli_print


class DragonFlyOptimizer(DiscreteSwarmOptimizer):

    def assertions(self):

        if self.velocity_clamp is not None:
            if not isinstance(self.velocity_clamp, tuple):
                raise TypeError("Parameter `velocity_clamp` must be a tuple")
            if not len(self.velocity_clamp) == 2:
                raise IndexError(
                    "Parameter `velocity_clamp` must be of " "size 2"
                )
            if not self.velocity_clamp[0] < self.velocity_clamp[1]:
                raise ValueError(
                    "Make sure that velocity_clamp is in the "
                    "form (v_min, v_max)"
                )

    def __init__(
            self,
            n_particles,
            dimensions,
            options={},
            init_pos=None,
            velocity_clamp=None,
            ftol=-np.inf,

    ):

        self.logger = logging.getLogger(__name__)
        super().__init__(
            n_particles=n_particles,
            dimensions=dimensions,
            options=options,
            binary=True,
            init_pos=init_pos,
            velocity_clamp=velocity_clamp,
            ftol=ftol,
        )

        self.food_fitness = np.inf
        self.enemy_fitness = -np.inf

        self.food_pos = np.empty(0)
        self.enemy_pos = np.empty(0)

        self.assertions()
        # Initialize the resettable attributes
        self.reset()
        # Initialize the topology
        self.top = Ring(static=False)

    def compute_pworst(self, swarm):  # Compute enemy position and cost
        try:
            # Infer dimensions from positions
            dimensions = swarm.dimensions
            # Create a 1-D and 2-D mask based from comparisons
            mask_cost = swarm.current_cost > swarm.pbest_cost
            mask_pos = np.repeat(mask_cost[:, np.newaxis], dimensions, axis=1)
            # Apply masks
            new_pworst_pos = np.where(~mask_pos, swarm.pbest_pos, swarm.position)
            new_pworst_cost = np.where(
                ~mask_cost, swarm.pbest_cost, swarm.current_cost
            )
        except AttributeError:
            msg = "Please pass a Swarm class. You passed {}".format(type(swarm))
            self.logger.error(msg)
            raise
        else:
            return (new_pworst_pos, new_pworst_cost)

    def _transfer_function(self, v):
        """Helper method for the transfer function

        Parameters
        ----------
        x : numpy.ndarray
            Input vector for sigmoid computation

        Returns
        -------
        numpy.ndarray

            Output transfer function computation
        """
        return abs(v / np.sqrt(v ** 2 + 1))

    def compute_position(self, velocity):
        return np.random.random_sample(size=self.dimensions) < self._transfer_function(velocity)

    def optimize(self, objective_func, iters, print_step=1, verbose=1, **kwargs):

        ub = 1
        lb = 0

        for i in range(iters):

            w = 0.9 - i * ((0.9 - 0.4) / iters)

            my_c = 0.1 - i * ((0.1 - 0) / (iters / 2))

            if my_c < 0:
                my_c = 0
            # print(my_c)
            s = 2 * random.random() * my_c  # Seperation weight
            a = 2 * random.random() * my_c  # Alignment weight
            c = 2 * random.random() * my_c  # Cohesion weight
            f = 2 * random.random()  # Food attraction weight
            e = my_c  # Enemy distraction weight

            # Compute cost for current position and personal best
            self.swarm.current_cost = objective_func(self.swarm.position, **kwargs)
            self.swarm.pbest_cost = objective_func(self.swarm.pbest_pos, **kwargs)
            self.swarm.pbest_pos, self.swarm.pbest_cost = compute_pbest(self.swarm)
            self.swarm.pworst_pos, self.swarm.pworst_cost = self.compute_pworst(self.swarm)

            pmin_cost_idx = np.argmin(self.swarm.pbest_cost)
            pmax_cost_idx = np.argmax(self.swarm.pworst_cost)
            # pmax_cost_idx = np.argmax(self.swarm.pbest_cost)
            # Update gbest from neighborhood

            # self.swarm.best_cost = np.min(self.swarm.pbest_cost)
            # self.swarm.pbest_pos = self.swarm.pbest_pos[np.argmin(self.swarm.pbest_cost)]

            # best_cost_yet_found = np.min(self.swarm.best_cost)

            self.swarm.best_pos, self.swarm.best_cost = self.top.compute_gbest(
                self.swarm, 2, self.n_particles
            )



            # Updating Food position
            if self.swarm.pbest_cost[pmin_cost_idx] < self.food_fitness:
                self.food_fitness = self.swarm.pbest_cost[pmin_cost_idx]
                self.food_pos = self.swarm.pbest_pos[pmin_cost_idx]

            # Updating Enemy position
            if self.swarm.pworst_cost[pmax_cost_idx] > self.enemy_fitness:
                self.enemy_fitness = self.swarm.pworst_cost[pmax_cost_idx]
                self.enemy_pos = self.swarm.pworst_pos[pmax_cost_idx]

            # best_cost_yet_found = np.min(self.swarm.best_cost)

            for j in range(self.n_particles):

                S = np.zeros(self.dimensions)
                A = np.zeros(self.dimensions)
                C = np.zeros(self.dimensions)
                F = np.zeros(self.dimensions)
                E = np.zeros(self.dimensions)

                # Calculating Separation(S)

                for k in range(self.n_particles):
                    S += (self.swarm.position[k] - self.swarm.position[j])

                S = -S

                # Calculating Allignment(A)

                for k in range(self.n_particles):
                    A += self.swarm.velocity[k]
                A = (A / self.n_particles)

                # Calculating Cohesion
                for k in range(self.n_particles):
                    C += self.swarm.position[k]
                C = (C / self.n_particles) - self.swarm.position[j]

                F = self.food_pos - self.swarm.position[j]  # Calculating Food postion
                E = self.enemy_pos - self.swarm.position[j]  # Calculating Enemy position

                self.swarm.velocity[j] = (s * S + a * A + c * C + f * F + e * E) + w * self.swarm.velocity[j]
                self.swarm.position[j] = self.compute_position(self.swarm.velocity[j])

            # Print to console
            if i % print_step == 0:
                cli_print(
                    "Iteration {}/{}, cost: {}".format(i + 1, iters, np.min(self.swarm.best_cost)),
                    verbose,
                    2,
                    logger=self.logger,
                )

        # Obtain the final best_cost and the final best_position
        # final_best_cost = np.min(self.swarm.pbest_cost)
        # final_best_pos = self.swarm.pbest_pos[np.argmin(self.swarm.pbest_cost)]

        final_best_cost = self.swarm.best_cost.copy()
        final_best_pos = self.swarm.best_pos.copy()

        print("==============================\nOptimization finished\n")
        print("Final Best Cost : ", final_best_cost, "\nBest Value : ", final_best_pos)

        # end_report(
        #     final_best_cost, final_best_pos, verbose, logger=self.logger
        # )
        return (final_best_cost, final_best_pos)
