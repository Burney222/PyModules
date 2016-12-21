from functools import wraps
import matplotlib.pyplot as plt
import matplotlib as mpl
import cycler  #color cycler
import gc_colors  #Update color palette

mpl.rcParams["axes.prop_cycle"] = cycler.cycler(color=["gcblue", "gcred", "gcgreen", "gcorange"])


def minorticks_decorate(func):
    @wraps(func)
    def func_wrapper(*args, **kwargs):
        output = func(*args, **kwargs)  #call the original function
        plt.minorticks_on()
        return output

    return func_wrapper

#Create new function
plot = minorticks_decorate(plt.plot)
