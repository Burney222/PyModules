from __future__ import division, print_function

import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import ROOT
import pandas as pd
import glob
import os
import re
from itertools import izip
import danplotlib as dpl

def SiPM2ASIC(channels):
    """Convert SiPM channel(array) into ASIC channel(array)"""
    return ((63-channels)%2) * 32 + (63-channels)//2

def ASIC2SiPM(channels):
    """Convert ASIC channel(array) into SiPM channel(array)"""
    return 63-((channels%32) * 2 + channels//32)

def SPIROC2Ths(channels, Ths=[1.5, 2.5, 4.5]):
    """Convert SPIROC pixel channel (or array or matrix) into PACIFIC like format
    (i.e. only values 0, 1, 2, 3 which correspond to the number of exceeded thresholds)"""
    if len(Ths) != 3:
        raise ValueError("ERROR: You must provide THREE thresholds!")

    x = channels
    return np.piecewise(x, [x < Ths[0], (x >= Ths[0]) & (x < Ths[1]), (x >= Ths[1]) & (x < Ths[2]),
                            x >= Ths[2]], [0, 1, 2, 3]).astype(int)

def extract_Ths(filename):
    #Check for naming Ths152535
    m = re.search("Ths(\d{6})", filename)
    if m:
        Ths_raw = m.group(1)
        return [ Ths_raw[i] + "." + Ths_raw[i+1] for i in [0, 2, 4] ]
    #Check for naming settingx.y
    m = re.search("setting\d.(\d)", filename)
    if m:
        ThSetting = m.group(1)
        if ThSetting == "1":
            return [ "0.5", "1.5", "2.5" ]
        if ThSetting == "2":
            return [ "1.5", "2.5", "3.5" ]
        if ThSetting == "3":
            return [ "1.5", "2.5", "4.5" ]

    return ["Low", "Mid", "High"]


def Files_DF(filepath):
    """Return pandas dataframe with files found in filepath. Filepath should contain wildcards"""
    DF_all_files = pd.DataFrame()
    DF_all_files["Filename"] = glob.glob(filepath)
    DF_all_files["Layer"] = [ int(re.search("Layer_(\d+)", filename).group(1)) for filename in DF_all_files["Filename"]]
    DF_all_files["Beamenergy"] = [ int(re.search("(\d+).0GeV", filename).group(1)) for filename in DF_all_files["Filename"]]
    DF_all_files["IntSetting"] = [ int(re.search("setting(\d+)", filename).group(1)) for filename in DF_all_files["Filename"]]
    DF_all_files["ThSetting"] = [ int(re.search("setting\d+.(\d+)", filename).group(1)) for filename in DF_all_files["Filename"]]
    DF_all_files["Position"] = [ int(re.search("(\d+)mm", filename).group(1)) for filename in DF_all_files["Filename"]]
    DF_all_files.loc[DF_all_files["Position"] == 150, "Position"] = 1500   #Correct position
    DF_all_files["Filename_short"] = [ os.path.basename(filename).replace("tb2017_1_pacific_", "")
                                      for filename in DF_all_files["Filename"]]

    return DF_all_files


def SPIROC_Files_DF(filepath):
    """Return pandas dataframe with files found in filepath. Filepath should contain wildcards"""
    DF_all_files = pd.DataFrame()
    DF_all_files["Filename"] = glob.glob(filepath)
    DF_all_files["Layer"] = [ int(re.search("Layer_(\d+)", filename).group(1)) for filename in DF_all_files["Filename"]]
    DF_all_files["Beamenergy"] = [ int(re.search("(\d+)GeV", filename).group(1)) for filename in DF_all_files["Filename"]]
    DF_all_files["ThSetting"] = [ int(re.search("ThS(\d)", filename).group(1))
                                 if re.search("ThS(\d)", filename) else -1
                                 for filename in DF_all_files["Filename"] ]
    DF_all_files["Filename_short"] = [ os.path.basename(filename) for filename in DF_all_files["Filename"]]

    return DF_all_files


def filter_files(Files_DF, settings):
    """Filter files_DF according to the settings specified in settings (dictionary)"""
    DF_selected_files = Files_DF
    if settings:
        for var, settings in settings.iteritems():
            if var == "inFilename": #Search for string in filename
                DF_selected_files = DF_selected_files.loc[ [settings in filename
                                                            for filename in DF_selected_files.loc[:,"Filename"]] ]
            else:  #Normal setting
                var_arr = DF_selected_files.loc[:,var]
                mask = [ False for i in range(len(var_arr))]
                for setting in settings:
                    mask |= (var_arr == setting)
                DF_selected_files = DF_selected_files.loc[ mask ]

    return DF_selected_files


#OUTDATED
def read_data(Files_DF, selection=None, clusteralgos=[], chunksize=None, columns=None):
    import root_pandas as rp
    """Read in data for given Files_DF and also read in clusters if applicable."""
    def data_and_clusters(filename):
        print("Reading in data from ", filename)
        data_DF = rp.read_root(filename, "PACIFIC", where=selection, chunksize=chunksize,
                               columns=columns)
        if chunksize:
            data_DF = data_DF.next()
        dirname, basename = os.path.split(filename)
        clusterfiles = glob.glob(dirname+"/Clusters/*/"+basename.replace(".root", ".h5"))
        print("Found clusterfiles: ", clusterfiles)
        for clusterfile in clusterfiles:
            algoname = re.search("Clusters/(\w+)/", clusterfile).group(1)
            if algoname in clusteralgos:
                print("Reading clusters for algorithm {} from {}".format(algoname, clusterfile))
                cluster_DF = pd.read_hdf(clusterfile)
                cluster_DF = cluster_DF.rename(index=str, columns={"Cluster" : "Clusters_{}".format(algoname)})
                print("Merge to existing Dataframe (column: Clusters_{})".format(algoname))
                events_before = len(data_DF)
                data_DF = pd.merge(data_DF, cluster_DF, how='inner', on=['Evt_num', 'BXing'])
                print("After merge {} events of {} remain.".format(len(data_DF), events_before))

        return data_DF

    return Files_DF.assign(Dataframe = lambda DF : [ data_and_clusters(filename)
                                                     for filename in DF.loc[:,"Filename"] ] )


def HDF2DF(filename, clusteralgos=["Default"], columns=None, BXs = None,
           ChData=False, ADCData=False, PixelData=False, selection=None,
           start=None, stop=None):
    """
    filename: Path of HDF file
    clusteralgos: List of cluster algorithms to read
    columns: List of columns to keep
    BXs: List of bunch crossings to read
    ch(ADC/Pixels)data: Boolean if to read these containers
    selection: Apply selection when reading (only works for data columns)
    start/stop: Specify lines/events to read
    """
    print("Reading in data from {}".format(filename))
    print("Cluster algorithms", clusteralgos)
    if BXs is not None:
        print("BXs", format(BXs))
        BXs = [ "BX{}".format(BX) for BX in BXs ]
    else:
        print("Read all BXs available (if applicable)")
        pat = re.compile("\/BX(-?\d+)\/Data")

    #Prepare keys to read
    readkeys = ["Data"]
    if ChData: readkeys.append("ChData")
    if ADCData: readkeys.append("ADCData")
    if PixelData: readkeys.append("PixelData")


    with pd.HDFStore(filename) as store:
        if BXs is None:
            keys = store.keys()
            BXs = [ int(pat.match(key).groups()[0]) for key in keys if pat.match(key) ]
            BXs = sorted(BXs)
            BXs = [ "BX{}".format(BX) for BX in BXs ]
            if BXs:
                print("Found BXs", BXs)
            else:
                print("No BXs found - assuming SPIROC data")
                BXs = [""]

        BX_DFs = []
        for BX in BXs:
            Key_DFs = []
            for readkey in readkeys:
                hdfkey = "{}/{}".format(BX, readkey)
                print("Reading {}...".format(hdfkey))
                where = selection if readkey=="Data" else None
                DF = store.select(hdfkey, where=where, start=start,
                                  stop=stop)
                Key_DFs.append(DF)

            for algo in clusteralgos:
                hdfkey = "{}/Clusters_{}".format(BX, algo)
                print("Reading {}...".format(hdfkey))
                DF = store.get(hdfkey)
                Key_DFs.append(DF)


            #Merge columns from different containers
            BX_DF = pd.concat(Key_DFs, axis=1, join="inner")
            #Select only subset of columns if applicable
            BX_DF = BX_DF[columns] if columns is not None else BX_DF

            BX_DFs.append( BX_DF ) #Append to list of BXs


    return pd.concat( BX_DFs, axis=0 ) #Merge and return BXs

#NEW VERSION OF read_data
def read_HDFs(Files_DF, selection=None, clusteralgos=[], columns=None):
    """Read in data for given Files_DF and also read in clusters if applicable."""

    return Files_DF.assign(Dataframe = lambda DF : [ HDF2DF(filename, clusteralgos, columns, selection)
                                                     for filename in DF.loc[:,"Filename"] ] )

def div0( a, b ):
    """ ignore / 0, div0( [-1, 0, 1], 0 ) -> [0, 0, 0] """
    with np.errstate(divide='ignore', invalid='ignore'):
        c = np.true_divide( a, b )
        c[ ~ np.isfinite( c )] = 0  # -inf inf NaN
    return c

def calc_TEfficiency(nPassed, nTotal):
    """Calculate efficiency using TEfficiency class from ROOT. nPassed and nTotal can be either a scalar each,
    or lists/arrays/... with the same length, where each entry correspond to a nPassed or nTotal respectively."""
    #Check for arrays input
    bScalar = True
    length = 1
    if hasattr(nPassed, "__iter__") or hasattr(nTotal, "__iter__"):  #Vector/list inputs
        if len(nPassed) != len(nTotal):
            raise ValueError("Inputs must have the same length")
        length = len(nPassed)
        bScalar = False
    else:  #If scalar for convenience put into a list with len 1
        nPassed = [ nPassed ]
        nTotal = [ nTotal ]

    #Create TEfficiency object and fill it
    TEff = ROOT.TEfficiency("Efficiency", "Efficiency", length, 0, length)
    for i in range(length):
        TEff.SetTotalEvents(i+1, nTotal[i])
        TEff.SetPassedEvents(i+1, nPassed[i])
    eff = np.asarray([ TEff.GetEfficiency(i+1) for i in range(length) ])
    eff_errlow = np.asarray([ TEff.GetEfficiencyErrorLow(i+1) for i in range(length) ])
    eff_errup = np.asarray([ TEff.GetEfficiencyErrorUp(i+1) for i in range(length) ])

    if bScalar:
        #Restore scalar behaviour
        eff = eff[0]
        eff_errlow = eff_errlow[0]
        eff_errup = eff_errup[0]

    return eff, eff_errlow, eff_errup



def calc_time(BX, Finetiming):
    """Calculate corresponding 'time' for given bunch crossing and Finetiming in signalshape plot"""
    return BX*32 + (31-Finetiming)




def finetimingplot(BX_arr, FT_arr, y_arr, BXs=[0,1,2], fig=None, axes=None,
                   newfigsize=(16,10), errbarsts = {}):
    """Generic function to plot the variable 'y_arr' (array) vs. BX and Finetiming on
    the x-axis.
    yerr will be passed to errorbar function as errors.
    BXs defines the BXs to actually plot
    You can pass already existing fig and axes to plot on top - or define the
    size of the new figure with newfigsize.
    errbarst: Settings passed to the errorbar calls
    """
    #Integrity checks
    if axes is not None:
        if fig is None:
            raise ValueError("Axes provided but no figure!")
        if len(axes) != len(BXs):
            raise ValueError("Length of axes must match length of BXs!")
    else:
        if fig is not None:
            raise ValueError("Figure provided but no axes!")
        print("Create new figure with sizes {}".format(newfigsize))
        fig, axes = dpl.subplots(1, len(BXs), sharey=True, figsize=newfigsize)

    #Check errorbar settings
    use_errbarsts = { "xerr" : 0.5, "yerr" : None, "label" : "" }
    #Override default settings
    use_errbarsts.update(errbarsts)

    #For handling the errors
    xerr = use_errbarsts["xerr"]
    yerr = use_errbarsts["yerr"]

    #Use plot instead of errorbar if both errors are set to 0
    use_plot = False
    if not xerr and not yerr:
        print("Errors set to 0/None, using mpl.plot instead of errorbar...")
        use_plot = True
        del use_errbarsts["xerr"]
        del use_errbarsts["yerr"]

    for idx, (axis, BX) in enumerate(zip(axes, BXs)):
        FT = FT_arr[ BX_arr == BX ]
        y = y_arr[ BX_arr == BX ]
        if yerr and not np.isscalar(yerr):
            #Common error provided?
            if len(yerr)==len(y_arr):
                yerr_sel = yerr[ BX_arr == BX ]
            #Asymmetric error?
            elif len(yerr) == 2:
                yerr_sel = [ yerr[0][ BX_arr == BX], yerr[1][ BX_arr == BX ] ]
            else:
                #Fall back to None
                print("Could not interpret yerr, set to None")
                yerr_sel = None
            #Update in dict
            use_errbarsts["yerr"] = yerr_sel


        if use_plot:
            axis.plot(FT, y, **use_errbarsts)
        else:
            axis.errorbar(FT, y, **use_errbarsts)
        axis.set_title("BX{}".format(BX))
        axis.set_xlim(-0.5, 31.5)
        axis.xaxis.set_ticks(np.arange(0,31,12))
        axis.minorticks_on()
        if idx != 0:
            axis.tick_params(axis="y", which="both", left="off")
        axis.invert_xaxis()

    fig.subplots_adjust(wspace=0)

    return fig, axes




def signalshape(BX_arr, Finetiming_arr, Ch_matrix, Th, mode="eq", binning=np.arange(0,8*32+1, 1)):
    """Function to plot the PACIFIC 'signalshape' for MULTIPLE CHANNELS.
    BX_arr, Finetiming_arr, Ch_arr must be arrays of the same length.
    Th is the threshold that shall be exceed (typically 1,2 or 3).
    """
    try:
        total_entries, n_channels = Ch_matrix.shape
    except ValueError:  #Array
        total_entries = len(Ch_matrix)
        n_channels=1

    if (len(BX_arr) != len(Finetiming_arr)) or (len(BX_arr) != total_entries):
        raise ValueError("Input arrays must have the same length.")

    #Calculate the "total" time
    #The finetiming needs to be "mirrored" because larger value means integrating only the left part
    time = calc_time(BX_arr, Finetiming_arr)

    #Select only entries that match the condition i.e. exceed the threshold
    if n_channels > 1:
        if mode=="eq":
            weights = np.sum( Ch_matrix==Th, axis=1, dtype=int )
        elif mode=="geq":
            weights = np.sum( Ch_matrix>=Th, axis=1, dtype=int )
        else:
            raise ValueError("'mode' must be either 'eq' (equal) or 'geq' (greater equal)")
    else:
        if mode=="eq":
            weights = np.asarray(Ch_matrix==Th, dtype=int).flatten()
        elif mode=="geq":
            weights = np.asarray(Ch_matrix>=Th, dtype=int).flatten()
        else:
            raise ValueError("'mode' must be either 'eq' (equal) or 'geq' (greater equal)")


    #Generate histograms
    time_hist, _ = np.histogram(time, bins=binning, weights=np.ones(len(time))*n_channels)
    time_th_hist, time_bins = np.histogram(time, bins=binning, weights=weights)

    #You need to divide the two histograms to correct for different trigger rates for different finetimings
    #And calculate the binomial error (you can treat total_hist as an efficiency...)
    total_hist = div0(time_th_hist, time_hist)
    total_hist_err = np.sqrt(np.abs(div0(total_hist*(1.-total_hist), time_hist)))

    return total_hist, time_bins, total_hist_err, total_hist_err



def signalshape_new(BX_arr, Finetiming_arr, Ch_matrix, Ths=[1,2,3], mode="eq", BXs=[0,1,2], cut_BX_val=None,
                    figsize=(16,10), ThNames=None, plotmean=False, figaxes=None, labels=None):
    """Function to plot the PACIFIC 'signalshape' for MULTIPLE CHANNELS.
    BX_arr, Finetiming_arr, Ch_arr must be arrays of the same length.
    Ths are the thresholds that shall be exceed (typically 1,2 or 3).
    """
    try:
        total_entries, n_channels = Ch_matrix.shape
    except ValueError:  #Array
        total_entries = len(Ch_matrix)
        n_channels=1

    if ThNames:
        if len(ThNames) != len(Ths):
            raise ValueError("Length of ThNames must match length of Ths!")
        else:
            ThNames_label = ThNames + ["\infty"]

    if figaxes is not None:
        if len(figaxes) != 2:
            raise ValueError("You must provide figaxes as a tuple of figure and axes: (fig, axes)")
        if len(figaxes[1]) != len(BXs):
            raise ValueError("Length of Axes must match length of BXs!")

    if (len(BX_arr) != len(Finetiming_arr)) or (len(BX_arr) != total_entries):
        raise ValueError("Input arrays must have the same length.")

    plotmode = mode
    if cut_BX_val:
        ref_BX = cut_BX_val[0]
        ref_val = cut_BX_val[1]

        if mode == "eq":
            mask = Ch_matrix[ BX_arr == ref_BX ] == ref_val
            plotmode = "eq"
        elif mode == "geq":
            mask = Ch_matrix[ BX_arr == ref_BX ] >= ref_val
            plotmode = "geq"
        elif mode == "eq|geq":
            mask = Ch_matrix[ BX_arr == ref_BX ] == ref_val
            plotmode = "geq"
        else:
            raise ValueError("'mode' must be either 'eq' (equal) or 'geq' (greater equal)")


    #Create new figure and axes if not available
    if figaxes is None:
        fig, axes = dpl.subplots(1, len(BXs), sharey=True, figsize=figsize)
    else:
        fig, axes = figaxes

    for axis, BX in zip(axes, BXs):
        weighted_mean_arr = []
        weighted_mean_arr_err = []
        for idx, Th in enumerate(Ths):
            finetimings = Finetiming_arr[ BX_arr == BX ]
            Chs = Ch_matrix[ BX_arr == BX ]

            #Cover all channels where not equal to ref_val in ref_BX
            if cut_BX_val:
                if mask.shape != Chs.shape:
                    raise ValueError("Ch_matrix shape between BXs {} and {} differ.".format(ref_BX, BX))
                Chs[~mask] = -1

            #Select only entries that match the condition i.e. exceed the threshold
            if n_channels > 1:
                if plotmode=="eq":
                    weights = np.sum( Chs==Th, axis=1, dtype=int )
                elif plotmode=="geq":
                    weights = np.sum( Chs>=Th, axis=1, dtype=int )
                else:
                    raise ValueError("'mode' must be either 'eq' (equal) or 'geq' (greater equal)")
            else:
                if plotmode=="eq":
                    weights = np.asarray(Chs==Th, dtype=int).flatten()
                elif plotmode=="geq":
                    weights = np.asarray(Chs>=Th, dtype=int).flatten()
                else:
                    raise ValueError("'mode' must be either 'eq' (equal) or 'geq' (greater equal)")


            #Generate histograms
            finetime_hist, _ = np.histogram(finetimings, bins=range(33), weights=np.ones(len(finetimings))*n_channels)
            finetime_th_hist, finetime_bins = np.histogram(finetimings, bins=range(33), weights=weights)

            #You need to divide the two histograms to correct for different trigger rates for different finetimings
            #And calculate the binomial error (you can treat total_hist as an efficiency...)
            total_hist = div0(finetime_th_hist, finetime_hist)
            weighted_mean_arr.append(total_hist)
            total_hist_err = np.sqrt(np.abs(div0(total_hist*(1.-total_hist), finetime_hist)))
            weighted_mean_arr_err.append(total_hist_err)

            #Manual labels
            if labels:
                label = labels[idx]

            #Automatic labels
            else:
                if plotmode == "geq":
                    if ThNames:
                        label = "[${},\infty$) pes".format(ThNames_label[idx])
                    else:
                        label = "[Th${},\infty$)".format(Th)
                elif plotmode == "eq":
                    if ThNames:
                        label = "[${},{}$) pes".format(ThNames_label[idx], ThNames_label[idx+1])
                    else:
                        label = "[Th${},$Th${}$)".format(Th, Th+1)

            axis.errorbar(np.arange(0.5, 32), total_hist, xerr=0.5, yerr=total_hist_err, fmt=".", ms=0,
                          label=label)

        #Plot weighted mean
        if plotmean:
            axis.errorbar(np.arange(0.5, 32), np.average(weighted_mean_arr, weights=Ths, axis=0), xerr=0.5,
                          yerr=np.average(weighted_mean_arr_err, weights=Ths, axis=0),
                          fmt=".", ms=0, label="Weighted mean", color="k")
        if figaxes is None:
            axis.set_title("BX{}".format(BX))
            axis.set_xlim(0,32)
            axis.xaxis.set_ticks(np.arange(0, 31, 12))
            axis.minorticks_on()
            if BX != BXs[0]:
                axis.tick_params(axis='y',          # changes apply to the x-axis
                                 which='both',      # both major and minor ticks are affected
                                 left="off")
            axis.invert_xaxis()

    if figaxes is None:
        axes[-1].set_xlabel("Finetiming")
        axes[0].set_ylabel("Fraction")
        fig.subplots_adjust(wspace=0)
        #plt.legend(fontsize=15)

    return fig, axes


def signalmean(BX_arr, Finetiming_arr, Ch_matrix, BXs=[0,1,2], figsize=(16,10)):
    """Function to plot the PACIFIC 'signalmean' for MULTIPLE CHANNELS.
    BX_arr, Finetiming_arr, Ch_arr must be arrays of the same length.
    """
    try:
        total_entries, n_channels = Ch_matrix.shape
    except ValueError:  #Array
        total_entries = len(Ch_matrix)
        n_channels=1

    if (len(BX_arr) != len(Finetiming_arr)) or (len(BX_arr) != total_entries):
        raise ValueError("Input arrays must have the same length.")


    fig, axes = dpl.subplots(1, len(BXs), sharey=True, figsize=figsize)
    for axis, BX in zip(axes, BXs):
        finetimings = Finetiming_arr[ BX_arr == BX ]
        Chs_BX = Ch_matrix[ BX_arr == BX ]

        FTs = range(32)
        means = []
        mean_errs = []
        for FT in FTs:
            Chs_FT = Chs_BX[ finetimings == FT ]

            mean, mean_err = masked_mean_error(Chs_FT)
            means.append(mean)
            mean_errs.append(mean_err)


        axis.errorbar(FTs, means, xerr=0.5, yerr=mean_errs,
                      fmt=".", ms=0)


        axis.set_title("BX{}".format(BX))
        axis.set_xlim(-0.5,31.5)
        axis.xaxis.set_ticks(np.arange(0, 31, 12))
        axis.minorticks_on()
        if BX != BXs[0]:
            axis.tick_params(axis='y',          # changes apply to the x-axis
                             which='both',      # both major and minor ticks are affected
                             left="off")
        axis.invert_xaxis()

    axes[-1].set_xlabel("Finetiming")
    axes[0].set_ylabel("Mean threshold")
    fig.subplots_adjust(wspace=0)
    #plt.legend(fontsize=16)

    return fig, axes



def masked_mean_error(arr, ignore_value=0, fill_value=None, axis=None):
    marr = np.ma.masked_array(arr, arr == ignore_value)
    mean = marr.mean(axis=axis)
    mean_err = stats.mstats.sem(marr, axis=axis)
    if axis != None and fill_value != None:
        mean = mean.filled(fill_value)
        mean_err = mean_err.filled(fill_value)

    return mean, mean_err




def plot_BX_overlay(axis=None):
    """Plot BX overlay on current plot.
    WARNING: You want to set the x-axis boundaries before calling this function!"""
    if axis:
        for bx in range(8+1):
            if( bx*32 < axis.get_xlim()[1] and bx*32 >= axis.get_xlim()[0]):
                axis.axvline(bx*32-0.5, lw=1, color="k")
                if(bx < 8 and (bx+0.7)*32 < axis.get_xlim()[1]):
                    axis.text(bx*32+32/2, axis.get_ylim()[1]*1.01, "BX{}".format(bx), ha="center")
    else:
        for bx in range(8+1):
            if( bx*32 < plt.xlim()[1] and bx*32 >= plt.xlim()[0]):
                plt.axvline(bx*32-0.5, lw=1, color="k")
                if(bx < 8 and (bx+0.7)*32 < plt.xlim()[1]):
                    plt.text(bx*32+32/2, plt.ylim()[1]*1.01, "BX{}".format(bx), ha="center")


def overthreshold_vs_channels(channel_matrix, Th, mode="ge", return_raw = False):
    """Calculate overthreshold ratios for multiple channels.
    If return_raw: Return raw number of events instead of the ratios.
    mode: ('ge', 'equal')"""
    total_entries, n_channels = channel_matrix.shape
    if mode == "ge":
        entries_over_th = np.sum( channel_matrix >= Th, axis=0 )
    elif mode == "le":
        entries_over_th = np.sum( channel_matrix <= Th, axis=0 )
    elif mode == "equal":
        entries_over_th = np.sum( channel_matrix == Th, axis=0 )
    else:
        raise ValueError("Mode must be 'ge' or 'equal'")

    if return_raw:
        return entries_over_th, range(n_channels), np.sqrt(entries_over_th)
    else:
        total_entries = len(entries_over_th) * [ total_entries ]
        ratio_over_th, ratio_over_th_errlow, ratio_over_th_errup = calc_TEfficiency(entries_over_th, total_entries)

        return ratio_over_th, range(n_channels), ratio_over_th_errlow, ratio_over_th_errup


def correlation(trackhit_arr, channel_matrix, Th, channel_range=range(64)):
    """Generate the correlation between a given track array and a channel matrix.
    The output are two arrays with the same length"""
    if (len(trackhit_arr) != len(channel_matrix)):
        raise ValueError("Inputs must have the same lengths (i.e. rows")

    channel_matrix = np.asarray(channel_matrix)  #Cast to array just in case
    total_entries, n_channels = channel_matrix.shape

    if n_channels != len(channel_range):
        raise ValueError("Provided channel numbers must match columns of channel_matrix!")

    channel_mask = (channel_matrix >= Th)

    Ch_over_th = []
    Corr_TrackHits = []

    for idx, ch in enumerate(channel_range):
        for trackhit in trackhit_arr[ channel_mask[:, idx] ]:
            Ch_over_th.append(ch)
            Corr_TrackHits.append(trackhit)

    return Corr_TrackHits, Ch_over_th



def cluster_correlation(trackhits, clusters, closest=False):
    """Generate the cluster correlation between a given track array and a list of lists of clusters.
    The output are two arrays with the same length, corresponding for the trackhit position and cluster position
    If closest=True, only return the cluster closest to the trackhit"""
    #Sanity checks
    if len(trackhits) != len(clusters):
        raise ValueError("Inputs must have the same lengths (i.e. rows")

    Cluster_positions = []
    Corr_TrackHits = []
    for trackhit, evt_clusters in izip(trackhits, clusters):
        trackhits_this = []
        clusterposis_this = []
        for cluster in evt_clusters:
            clusterposis_this.append(cluster.Position())
            trackhits_this.append(trackhit)

        #Filter if closest
        if closest and len(trackhits_this)>0:
            absdiff = [ abs(hit-cluster) for hit, cluster
                       in zip(trackhits_this, clusterposis_this) ]
            idxmin = np.argmin(absdiff)
            trackhits_this = [ trackhits_this[idxmin] ]
            clusterposis_this = [ clusterposis_this[idxmin] ]

        Cluster_positions.extend(clusterposis_this)
        Corr_TrackHits.extend(trackhits_this)

    return Corr_TrackHits, Cluster_positions



def cluster_efficiencies(trackhits, clusters, resolution=4, channel_range=range(64), return_raw=False):
    """Calculate per channel efficiencies. Resolution is given in channel-units.
    - trackhits is a list/tuple/.. of trackhit positions (in channel units).
    - clusters is a list of lists of generated clusters.
    - channel_range is an array with the channels correpsonding to the channels for which the clusters were build.
      Does only really makes sense if channel_range is a connected series..
    - resolution tells the area around trackhit position to look for a cluster.
    - If return_raw is set to true, return the number of found and total trackhits instead of
      returning the efficiency directly (might be useful when only calculating the efficiency for sub-range
      of channels.)"""
    #Sanity checks
    if len(trackhits) != len(clusters):
        raise ValueError("Trackhit array and cluster array must be of the same length!")

    #Quantities for the efficiency
    total_trackhits = np.zeros(len(channel_range), dtype=np.int64)
    found_trackhits = np.zeros(len(channel_range), dtype=np.int64)

    #Quantities for the purity
    total_clusters = np.zeros(len(channel_range), dtype=np.int64)
    found_clusters = np.zeros(len(channel_range), dtype=np.int64)
    for trackhit, evt_clusters in izip(trackhits, clusters):
        #Rounded trackposition
        track_ch = int(round(trackhit))
        #Trackhit falls into the channel range
        if track_ch in channel_range:
            idx = channel_range.index(track_ch)
            total_trackhits[idx] += 1
            bTrackhit_found = False
            for cluster in evt_clusters:
                cluster_posi = cluster.Position()
                cluster_ch = int(round(cluster_posi))
                if cluster_ch in channel_range:
                    #Cluster idx needed for the purity
                    cluster_idx = channel_range.index( cluster_ch )
                else:
                    cluster_idx = None
                if abs(trackhit-cluster_posi) < resolution:
                    #Found first cluster that belongs to the trackhit
                    if not bTrackhit_found:
                        found_trackhits[idx] += 1
                        bTrackhit_found = True
                    #In any case handle the found clusters for the purity (if cluster in chanrange)
                    if cluster_idx:
                        found_clusters[cluster_idx] += 1
                if cluster_idx:
                    total_clusters[cluster_idx] += 1

        #Trackhit does not fall into the channel range
        #Still can do purity
        else:
            for cluster in evt_clusters:
                cluster_posi = cluster.Position()
                cluster_ch = int(round(cluster_posi))
                if cluster_ch in channel_range:
                    cluster_idx = channel_range.index( cluster_ch )
                    if abs(trackhit-cluster_posi) < resolution:
                        found_clusters[cluster_idx] += 1
                    total_clusters[cluster_idx] += 1

    if return_raw:
        return found_trackhits, total_trackhits, found_clusters, total_clusters, channel_range

    else:
        channel_eff = calc_TEfficiency(found_trackhits, total_trackhits)   #Efficiencies per channel
        total_eff = calc_TEfficiency(np.sum(found_trackhits), np.sum(total_trackhits)) #Total efficiency

        channel_pur = calc_TEfficiency(found_clusters, total_clusters)  #Purity per channel
        total_pur = calc_TEfficiency(np.sum(found_clusters), np.sum(total_clusters))  #Total purity

        return total_eff, channel_eff, total_pur, channel_pur, channel_range
