#!/usr/bin/env python

##
#
# This file contains code to obtain posterior inference for a variety of
# candidate likelihood functions. These likelihood functions describe the
# distribution from which SLAM observations (x,y,z position) are drawn.
#
##

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import invwishart, multivariate_normal, pareto, uniform

class NoiseModel:
    """
    This is a superclass for objects that describe different noise
    models.
    """
    def __init__(self):
        # This parameter is set to False by default, and changes once
        # we get some data and perform poterior inference over any unknown
        # parameters. 
        self.has_inference = False

    def posterior_inference(self, ground_truth_data, estimate_data):
        """
        Find the posterior values of any unknown parameters given the data
        """
        self.has_inference = True
        pass

    def simulate_run(self, ground_truth):
        """
        Given a true sequence of positions, simulate what we think 
        the SLAM system might output.
        """
        assert self.has_inference , "No posterior inference available yet!"

        simulated_errors = self.simulate_error(len(ground_truth.T))
        simulated_run = ground_truth + simulated_errors

        return simulated_run

    def simulate_error(self, run_len):
        """
        Simulate errors from the actual observation according to this noise model
        """
        assert self.has_inference , "No posterior inference available yet!"
        pass

    def simulate_error_set(self, num_runs, run_len):
        """
        Generate a list of simulated errors. Each error run has length (run_len),
        and there are (num_runs) runs in total
        """
        assert self.has_inference , "No posterior inference available yet!"
        runs = []

        for i in range(num_runs):
            run = self.simulate_error(run_len)
            runs.append(run)

        return runs

class NormalLikelihood(NoiseModel):
    """
    Assumes that SLAM observations are drawn from a normal distribution
    with mean = actual position and unknown variance.
    """
    def posterior_inference(self, ground_truth_data, estimate_data):    
        """
        Prior: Inverse-Wishart with d+1=4 degrees of freedom.
        This gives uniform marginal priors on pairwise correlations between variables (Gelman p.73)

        See https://en.wikipedia.org/wiki/Inverse-Wishart_distribution#Conjugate_distribution for
        derivation of the posterior.
        """
        assert ground_truth_data.shape == estimate_data.shape , "Ground truth and estimate shapes don't match!"
        error = ground_truth_data - estimate_data

        # Prior
        v = 4               # degrees of freedom
        W = np.identity(3)  # scaling factor

        # Posterior
        n = len(error.T)
        A = error * error.T
        v_post = v + n
        W_post = W + A

        # Generate a scipy random variable object.
        self.posterior = invwishart(v_post,W_post)

        self.has_inference = True

    def simulate_error(self, n):
        """
        Simulate errors from the actual observation according to this noise model
        """
        assert self.has_inference , "No posterior inference available yet!"
       
        simulated_error = np.full((3,n), np.inf)

        # Make n posterior samples of the variance
        Sigmas = self.posterior.rvs(size=n)

        # Use these samples to sample from the noise distribution (multivariate gaussian)
        for i in range(n):
            Sigma = Sigmas[i]
            simulated_error[:,i] = multivariate_normal.rvs(np.zeros(3), Sigma)

        return simulated_error

class UniformLikelihood(NoiseModel):
    """
    Assumes that SLAM observations in each axis are drawn from a uniform distribution
    centered on the actual value with unknown support.
    """
    def posterior_inference(self, ground_truth_data, estimate_data):    

        assert ground_truth_data.shape == estimate_data.shape , "Ground truth and estimate shapes don't match!"
        error = np.asarray(ground_truth_data - estimate_data)
        n = len(error.T)

        # Prior: Pareto distribution with 
        # shape parameter k=0.1 and scale parameter v=0.1
        k = 0.01
        v = 0.01

        # Posterior: Pareto distribution with updated shape and scale parameters.
        # we have different scale parameters for x, y, z.
        k_post = k + n

        v_post_x = max( v, max(abs(error[0,:])) )
        v_post_y = max( v, max(abs(error[1,:])) )
        v_post_z = max( v, max(abs(error[2,:])) )

        # Generate scipy random variable objects
        self.posterior_x = pareto(k_post,loc=0,scale=v_post_x)
        self.posterior_y = pareto(k_post,loc=0,scale=v_post_y)
        self.posterior_z = pareto(k_post,loc=0,scale=v_post_z)

        self.has_inference = True

    def simulate_error(self, n):
        """
        Simulate errors from the actual observation according to this noise model
        """
        assert self.has_inference , "No posterior inference available yet!"
       
        simulated_error = np.full((3,n), np.inf)

        # Draw posterior samples of the bounds on the uniform distribution
        x_bounds = self.posterior_x.rvs(size=n)
        y_bounds = self.posterior_y.rvs(size=n)
        z_bounds = self.posterior_z.rvs(size=n)

        # Use these samples to draw from the noise distribution (uniform)
        for i in range(n):
            x_bound = x_bounds[i]
            y_bound = y_bounds[i]
            z_bound = z_bounds[i]

            x_err = uniform.rvs(loc=(-x_bound), scale=(2*x_bound))
            y_err = uniform.rvs(loc=(-y_bound), scale=(2*y_bound))
            z_err = uniform.rvs(loc=(-z_bound), scale=(2*z_bound))

            simulated_error[:,i] = np.array([x_err,y_err,z_err])

        return simulated_error
    
