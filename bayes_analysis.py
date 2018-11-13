#!/usr/bin/env python

##
#
# A script to provide bayesian analysis of the uncertainty in SLAM localization
#
##

import csv
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import invwishart, multivariate_normal

# this directory contains tools from the ros package for postprocessing data
benchmark_dir = "/home/vjkurtz/catkin_ws/src/rgbdslam_v2/rgbd_benchmark"
sys.path.insert(0,benchmark_dir)

# import tools for postprocessing (from cython library evaluate_ate_module.pyx)
import pyximport; pyximport.install()
from evaluate_ate_module import *     

# Location of data we'll work with
data_dir = "/home/vjkurtz/catkin_ws/src/rgbdslam_v2/test/2018-11-08_16:46/emm__0.00/CANDIDATES_4/RANSAC_100/OPT_SKIP_10/ORB/600_Features/rgbd_dataset_freiburg1_room"
ground_truth_file = data_dir + "/rgbd_dataset_freiburg1_room-groundtruth.txt"
estimate_file = data_dir + "/rgbd_dataset_freiburg1_room.bagiteration_1_estimate.txt"

def preprocess(first_file, second_file):
    """
    Given two datafiles, process them to extract (x,y,z) trajectories
    in the reference frame of the first file. 

    returns:
        - first_xyz : the first trajectory
        - second_xyz_aligned : the aligned second trajectory
    """
    first_list = associate.read_file_list(first_file)
    second_list = associate.read_file_list(second_file)

    # miscellaneous options
    scale = 1.0             # scaling factor for second trajectory
    offset = 0.0            # time offset added to second file
    max_difference = 0.02   # maximum time difference for corresponding entries

    # Match up points in the two trajectories
    matches = associate.associate(first_list, second_list, offset, max_difference)
    assert len(matches) > 2 , "Insufficient matches. Are these really correponding sequences?"

    # get matrices of x,y,z position
    first_xyz = np.matrix([[float(value) for value in first_list[a][0:3]] for a,b in matches]).transpose()
    second_xyz = np.matrix([[float(value)*scale for value in second_list[b][0:3]] for a,b in matches]).transpose()

    # Find transformation between first and second trajectories
    rot, trans, trans_error = align(second_xyz, first_xyz)
    second_xyz_aligned = rot * second_xyz + trans

    return first_xyz, second_xyz_aligned

def posterior_inference(error):
    """
    Assume that erros are drawn from a gaussian distribution with zero mean. 
    Find posterior inference over the variance.

    returns a scipy.stats random variable object
    """

    # Prior on Sigma: Inverse-Wishart with d+1=4 degrees of freedom
    #   this gives uniform marginal prior on pairwise correlations (see Gelman p 73)
    v = 4
    W = np.identity(3)

    # Posterior on Sigma: see https://en.wikipedia.org/wiki/Inverse-Wishart_distribution#Conjugate_distribution
    n = len(error.T)
    A = error * error.T
    v_post = v + n
    W_post = W + A

    posterior = invwishart(v_post, W_post)

    return posterior

def simulate_run(posterior, ground_truth):
    """
    Given a posterior distribution over the error and the actual value, 
    generate a simulated run.
    """
    n = len(ground_truth.T)

    simulated_run = np.full(ground_truth.shape, np.inf)

    for i in range(n):
        # sample the variance
        Sigma = posterior.rvs(size=1)

        # sample a disturbance from the ground truth mean
        dist = multivariate_normal.rvs(np.zeros(3),Sigma)

        # apply this disturbance to calculate the real trajectory
        simulated_run[:,i] = ground_truth.T[i] + dist

    return simulated_run

def simulate_error(posterior, n):
    """
    Generate a sequence of errors with length n, drawing from the
    given posterior distribution.
    """

    simulated_error = np.full((3,n), np.inf)

    for i in range(n):
        # sample the variance
        Sigma = posterior.rvs(size=1)

        # sample from the error distribution
        simulated_error[:,i] = multivariate_normal.rvs(np.zeros(3), Sigma)

    return simulated_error


def simulate_errors(posterior, num_runs, len_runs):
    """
    Given a posterior distirbution over variance, this function generates a 
    list of simulated errors. Each run has length (len_runs) and there are
    (num_runs) in total.
    """
    runs = [] 
    
    for i in range(num_runs):
        run = simulate_error(posterior, len_runs)
        runs.append(run)

    return runs

def plot_errors(actual_error, simulated_error):
    """
    Given an actual run and one generated via simulation, plot the errors
    of each. This will give a qualitative sense of whether the noise model 
    makes sense.
    """
    plt.subplot(2,1,1)
    plt.plot(actual_error.T)
    plt.xlabel("timestep")
    plt.ylabel("tracking error")
    plt.title("Actual Error")

    plt.subplot(2,1,2)
    plt.plot(simulated_error.T)
    plt.title("Simulated Error")
    plt.xlabel("timestep")
    plt.ylabel("tracking error")

    plt.show()

def plot_residuals(ground_truth, actual_run, posterior):
    """
    Given posterior inference over the errors from the actual run
    and a ground truth run, plot the bayesian residuals of the estimated
    error.
    """
    pass

def bayesian_p_value(actual_error, simulated_errors, T, plot=True, title="P Value"):
    """
    Given a run of actual values, a list of similar simulated values
    and a test statistic, calculate the Bayesian P value. If requested, 
    plot a corresponding histogram.

    Arguments:
        actual    : a (d x n) np array of actual recorded values
        simulated : a list of M (d x n) np arrays of simulated values
        T         : a function mapping a (d x n) np array to a single real value
        plot      : [optional] a boolean value indicating whether to plot a histogram
        title     : [optional] a title for the plot

    Returns:
        p         : a value in [0,1], the Bayesian p-value for the given test statistic

    """

    # define the test statistic
    #T = lambda x : np.mean(x, axis=1)

    # Generate a bunch of sample runs and calculate their statistic
    stats = []

    for run in simulated_errors:
        stats.append( T(run) )


    # Calculate the actual value of the statistic
    actual_stat = T(actual_error)

    # figure out the proportion of simulated stats that are above the actual stat
    n_above = sum( i > actual_stat for i in stats )
    p = float(n_above) / float(len(simulated_errors))

    # Create a histogram of this statistic
    if plot:
        plt.hist(stats, label="simulated data")
        plt.axvline(actual_stat, color="red", label="actual value")
        plt.title(title + " P = %s" % (p))
        plt.xlabel("Value of Test Statistic")
        plt.ylabel("Frequency")
        plt.legend()

    return p




if __name__=="__main__":
    print("===> Loading Data")
    ground_truth, estimate = preprocess(ground_truth_file, estimate_file)
    actual_error = ground_truth - estimate

    print("===> Performing Posterior Inference")
    post = posterior_inference(actual_error)

    print("===> Simulating Errors")
    sim_errors = simulate_errors(post, 50, len(actual_error.T))

    print("===> Calculating Bayesian P-Value")
    T = np.min
    bayesian_p_value(actual_error, sim_errors, T)
    plt.show()

    plot_errors(actual_error, sim_errors[0])


