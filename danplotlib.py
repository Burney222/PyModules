"""
Module for convenient plotting using matplotlib.pyplot
"""
from functools import wraps, partial, update_wrapper
import matplotlib.pyplot as plt
import matplotlib as mpl
import cycler  #color cycler
import gc_colors  #Update color palette

#Update default color palette
plt.rcParams["axes.prop_cycle"] = cycler.cycler(color=["gcblue", "gcred", "gcgreen", "gcorange"])


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
