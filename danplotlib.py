"""
Module for convenient plotting using matplotlib.pyplot
"""
from __future__ import division, print_function
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

            #Set default linewidth and hist type for histogram plots
            axis_hist_temp = axis.hist
            axis.hist = partial(axis_hist_temp, linewidth=plt.rcParams["lines.linewidth"],
                                histtype="step")
            update_wrapper(axis.hist, axis_hist_temp)


            #Set default errorbar format and ms
            axis_errorbar_temp = axis.errorbar
            axis.errorbar = partial(axis_errorbar_temp, fmt=".", ms=0)
            update_wrapper(axis.errorbar, axis_errorbar_temp)

        if to_list: #Restore old behaviour
            ax = ax[0]

        return f, ax
    return func_wrapper

subplots = trafo_subplots_axes(plt.subplots)





#=================================================================
# Add additional functionalities =================================
#=================================================================
def div0( a, b ):
    """ ignore / 0, div0( [-1, 0, 1], 0 ) -> [0, 0, 0] """
    with np.errstate(divide='ignore', invalid='ignore'):
        c = np.true_divide( a, b )
        c[ ~ np.isfinite( c )] = 0  # -inf inf NaN
    return c

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

    modified_z_score = div0(0.6745 * diff, med_abs_deviation)

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
            data_x = line.get_xdata()
            data_y = line.get_ydata()

            #In case of numpy datetime64 -> convert to datetime.datetime
            if isinstance(data_x[0], np.datetime64):
                data_x = data_x.astype("M8[ms]").astype("O")
            if isinstance(data_y[0], np.datetime64):
                data_y = data_y.astype("M8[ms]").astype("O")

            #Convert to float in case of datetime.datetime
            try:
                data_x = mpl.dates.date2num(data_x)
            except AttributeError:
                data_x = data_x

            try:
                data_y = mpl.dates.date2num(data_y)
            except AttributeError:
                data_y = data_y


            data = data_x if xory == "x" else data_y
            #Ignore axvline, axhline
            if not isinstance(data, list):
                #Only consider data in current view (e.g. if did xlim before)
                #X-direction
                xlims = axis.get_xlim()
                viewmask = (data_x >= xlims[0]) & (data_x <= xlims[1])
                #Y-direction
                ylims = axis.get_ylim()
                viewmask &= (data_y >= ylims[0]) & (data_y <= ylims[1])

                data_view = data[viewmask]

                #Remove outliers
                if thresh == 0: #Do not remove outliers
                    data_filtered = data_view
                else:
                    data_filtered = data_view[~is_outlier(data_view, thresh)]
                up = np.nanmax( [np.nanmax(data_filtered), up] )
                low = np.nanmin( [np.nanmin(data_filtered), low] )

        if xory == "x":
            axis.set_xlim(low-extend_view*(up-low), up+extend_view*(up-low))
        else:
            axis.set_ylim(low-extend_view*(up-low), up+extend_view*(up-low))


#Patch matplotlib Axes
mpl.axes.Axes.autoscale_nooutliers = autoscale_nooutliers
