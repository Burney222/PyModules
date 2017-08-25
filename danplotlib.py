"""
Module for convenient plotting using matplotlib.pyplot
"""
from functools import wraps, partial, update_wrapper
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np


#Define new plot functions which also plot minorticks
def minorticks_decorate(func):
    @wraps(func)
    def func_wrapper(*args, **kwargs):
        output = func(*args, **kwargs)  #call the original function
        plt.minorticks_on()
        return output

    return func_wrapper

plot = minorticks_decorate(plt.plot)
step = minorticks_decorate(plt.step)
errorbar_temp = minorticks_decorate(plt.errorbar)
hist_temp = minorticks_decorate(plt.hist)

#Change default linewidth for histograms to global default
hist = partial(hist_temp, linewidth=plt.rcParams["lines.linewidth"], histtype="step")
update_wrapper(hist, hist_temp)

#Change default appearance for errorbar
errorbar = partial(errorbar_temp, fmt=".", ms=0)
update_wrapper(errorbar, errorbar_temp)

#Change default position for xlabel and ylabel
xlabel = partial(plt.xlabel, ha="right", x=1)
update_wrapper(xlabel, plt.xlabel)
ylabel = partial(plt.ylabel, ha="right", y=1)
update_wrapper(ylabel,plt.ylabel)


#Do the same things for the axes-plotting commands
#Define own Axes class
class Axes(mpl.axes.Axes):
    plot = minorticks_decorate(mpl.axes.Axes.plot)

def trafo_subplots_axes(func):
    @wraps(func)
    def func_wrapper(*args, **kwargs):
        f, ax = func(*args, **kwargs)  #call the original function
        #ax might be single axis object or array of axis objects
        to_list = False   #If artificially cast to list
        if not hasattr(ax, '__iter__'):
            ax = [ ax ]
            to_list = True

        for axis in ax:  #Do the update of the methods
            axis.minorticks_on() #Turn on minorticks
            #Adjust default xlabel position
            set_xlabel_temp = axis.set_xlabel
            axis.set_xlabel = partial(set_xlabel_temp, ha="right", x=1)
            update_wrapper(axis.set_xlabel, set_xlabel_temp)
            #Adjust default ylabel position
            set_ylabel_temp = axis.set_ylabel
            axis.set_ylabel = partial(set_ylabel_temp, ha="right", y=1)
            update_wrapper(axis.set_ylabel, set_ylabel_temp)

        if to_list: #Restore old behaviour
            ax = ax[0]

        return f, ax
    return func_wrapper

subplots = trafo_subplots_axes(plt.subplots)





#=================================================================
# Add additional functionalities =================================
#=================================================================

def is_outlier(points, thresh=3.5):
    """
    Returns a boolean array with True if points are outliers and False
    otherwise.

    Parameters:
    -----------
        points : An numobservations by numdimensions array of observations
        thresh : The modified z-score to use as a threshold. Observations with
            a modified z-score (based on the median absolute deviation) greater
            than this value will be classified as outliers.

    Returns:
    --------
        mask : A numobservations-length boolean array.

    References:
    ----------
        Boris Iglewicz and David Hoaglin (1993), "Volume 16: How to Detect and
        Handle Outliers", The ASQC Basic References in Quality Control:
        Statistical Techniques, Edward F. Mykytka, Ph.D., Editor.
    """
    if len(points.shape) == 1:
        points = points[:,None]
    median = np.median(points, axis=0)
    diff = np.sum((points - median)**2, axis=-1)
    diff = np.sqrt(diff)
    med_abs_deviation = np.median(diff)

    modified_z_score = 0.6745 * diff / med_abs_deviation

    return modified_z_score > thresh

#Function to adjust scaling, ignoring outliers
def autoscale_nooutliers(axis=None, direction="y", extend_view=0.1, thresh=3.5):
    if not axis:
        axis=plt.gca()

    if direction not in ["x", "y", "both"]:
        raise ValueError("direction must be either 'x', 'y' or 'both'!")
    elif direction == "both":
        direction = ["x", "y"]
    else:
        direction = [ direction ]

    #Loop through directions
    for xory in direction:
        low = np.nan
        up = np.nan
        #Loop through line objects of axis
        for line in axis.lines:
            data = line.get_xdata() if xory == "x" else line.get_ydata()
            if not isinstance(data, list): #Ignore axvline, axhline
                data_filtered = data[~is_outlier(data, thresh)]
                up = np.nanmax( [np.nanmax(data_filtered), up] )
                low = np.nanmin( [np.nanmin(data_filtered), low] )

        if xory == "x":
            axis.set_xlim(low-extend_view*(up-low), up+extend_view*(up-low))
        else:
            axis.set_ylim(low-extend_view*(up-low), up+extend_view*(up-low))


#Patch matplotlib Axes
mpl.axes.Axes.autoscale_nooutliers = autoscale_nooutliers
