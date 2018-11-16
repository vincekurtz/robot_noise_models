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
from statsmodels.tsa.stattools import acf
from noise_models import *

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

def plot_errors(actual_error, simulated_error):
    """
    Given an actual run and one generated via simulation, plot the errors
    of each. This will give a qualitative sense of whether the noise model 
    makes sense.
    """
    fig, (ax1, ax2) = plt.subplots(2,1,sharey=True)

    ax1.plot(actual_error.T)
    #ax1.set_xlabel("timestep")
    ax1.set_ylabel("tracking error")
    ax1.set_title("Actual Error")

    ax2.plot(simulated_error.T)
    ax2.set_xlabel("timestep")
    ax2.set_ylabel("tracking error")
    ax2.set_title("Simulated Error")

def plot_residuals(actual, simulated):
    """
    Given posterior inference over the errors from the actual run
    and a ground truth run, plot the bayesian residuals of the estimated
    error.
    """
    pass

def bayesian_p_value(actual, simulated, T, plot=False, title=""):
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

    for run in simulated:
        stats.append( T(run) )


    # Calculate the actual value of the statistic
    actual_stat = T(actual)

    # figure out the proportion of simulated stats that are above the actual stat
    n_above = sum( i > actual_stat for i in stats )
    p = float(n_above) / float(len(simulated))

    # Create a histogram of this statistic
    if plot:
        plt.hist(stats, label="simulated data")
        plt.axvline(actual_stat, color="red", label="actual value")
        plt.title(title + " (p=%s)" % (p))
        plt.xlabel("Value of Test Statistic")
        plt.ylabel("Frequency")
        plt.legend()

    return p

def plot_p_values(actual_error, simulated_errors, axis=0):
    """
    Make a cute plot of Bayesian P-values and histograms for
    a variety of test statistics on the given axis.

    Statistics used:
        - min/max
        - quantile
        - mean
        - median
        - variance
        - autocorrelation
    """
    plt.figure()

    # Just consider error on the given axis, which we'll call 'x' w.l.o.g.
    actual_error_x = np.asarray(actual_error[axis,:])   # convert to np array since matrix messes things up
    simulated_errors_x = [ np.asarray(sim_err[axis,:]) for sim_err in simulated_errors ]

    # Minimum
    plt.subplot(2,4,1)
    bayesian_p_value(actual_error_x, simulated_errors_x, np.min, plot=True, title="Minimum")

    # Maximum
    plt.subplot(2,4,2)
    bayesian_p_value(actual_error_x, simulated_errors_x, np.max, plot=True, title="Maximum")

    # 10% quantile
    plt.subplot(2,4,3)
    low_quant = lambda x : np.percentile(x, 10.0)
    bayesian_p_value(actual_error_x, simulated_errors_x, low_quant, plot=True, title="10% Quantile")

    # 90% quanitle
    plt.subplot(2,4,4)
    high_quant = lambda x : np.percentile(x, 90.0)
    bayesian_p_value(actual_error_x, simulated_errors_x, high_quant, plot=True, title="90% Quantile")

    # Mean
    plt.subplot(2,4,5)
    bayesian_p_value(actual_error_x, simulated_errors_x, np.mean, plot=True, title="Mean")

    # Median
    plt.subplot(2,4,6)
    bayesian_p_value(actual_error_x, simulated_errors_x, np.median, plot=True, title="Median")

    # Autocorrelation with lag t=10
    plt.subplot(2,4,7)
    t = 10
    auto_corr = lambda x : acf(x,nlags=t)[t]
    bayesian_p_value(actual_error_x, simulated_errors_x, auto_corr, plot=True, title="Autocorrelation with lag 10")

    # Variance
    plt.subplot(2,4,8)
    bayesian_p_value(actual_error_x, simulated_errors_x, np.var, plot=True, title="Variance")

def summary_plots(noise_model, n_samples=50):
    """
    Given a noise model object (from noise_models.py), calculate and show some summary plots.
    """
    print("===> Loading Data")
    ground_truth, estimate = preprocess(ground_truth_file, estimate_file)
    actual_error = ground_truth - estimate

    # Set up a simulator that assumes a normal likelihood function
    print("===> Getting Posterior Inference")
    noise_model.posterior_inference(ground_truth, estimate)

    # Simulate a bunch of errors
    print("===> Simulating Error Set")
    sim_errors = noise_model.simulate_error_set(n_samples,len(ground_truth.T))

    # Compare these to the actual error
    print("===> Calculating P-Values")
    plot_p_values(actual_error, sim_errors, axis=0)

    plot_errors(actual_error, sim_errors[0])

    print("===> Displaying Plots")
    plt.show()


if __name__=="__main__":
    normal_sim = NormalLikelihood()
    uniform_sim = UniformLikelihood()

    gp_sim = GaussianProcessLikelihood()
    summary_plots(gp_sim, n_samples=1)

    #summary_plots(uniform_sim)
    #summary_plots(normal_sim)

