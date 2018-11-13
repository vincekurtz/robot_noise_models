#!/usr/bin/env python

##
#
# This file contains code to obtain posterior inference for a variety of
# candidate likelihood functions. These likelihood functions describe the
# distribution from which SLAM observations (x,y,z position) are drawn.
#
##

import numpy as np
from scipy.stats import invwishart, multivariate_normal

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

    
