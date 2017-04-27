"""
Module for convenient plotting using matplotlib.pyplot
"""
from functools import wraps, partial, update_wrapper
import matplotlib.pyplot as plt
import matplotlib as mpl
import gc_colors  #Update color palette


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
hist = partial(hist_temp, linewidth=plt.rcParams["lines.linewidth"])
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
