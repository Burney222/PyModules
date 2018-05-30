from __future__ import division
import numpy as np
import ROOT

def calc_TEfficiency(nPassed, nTotal, TEff = True):
    """Calculate efficiency using TEfficiency class from ROOT (default) or simple method.
    nPassed and nTotal can be either a scalar each, or lists/arrays/... with the same length,
    where each entry correspond to a nPassed or nTotal respectively.
    Returns (eff, eff_errlow, eff_errup)"""

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

    if(TEff):
        #Create TEfficiency object and fill it
        HistPassed = ROOT.TH1D("Passed", "Passed", length, 0, length)
        HistTotal = ROOT.TH1D("Total", "Total", length, 0, length)

        for i in range(length):
            HistTotal.SetBinContent(i+1, nTotal[i])
            HistPassed.SetBinContent(i+1, nPassed[i])

        TEff = ROOT.TEfficiency(HistPassed, HistTotal)
        TEff.SetUseWeightedEvents(False) #Otherwise errors are not computed..

        eff = np.asarray([ TEff.GetEfficiency(i+1) for i in range(length) ])
        eff_errlow = np.asarray([ TEff.GetEfficiencyErrorLow(i+1) for i in range(length) ])
        eff_errup = np.asarray([ TEff.GetEfficiencyErrorUp(i+1) for i in range(length) ])

    else:
        nPassed = np.asarray(nPassed, dtype=np.float64)
        nTotal = np.asarray(nTotal, dtype=np.float64)
        eff = nPassed/nTotal
        eff_errlow = np.sqrt(eff*(1.-eff)/nTotal)
        eff_errup = eff_errlow

    if bScalar:
        #Restore scalar behaviour
        eff = eff[0]
        eff_errlow = eff_errlow[0]
        eff_errup = eff_errup[0]

    return eff, eff_errlow, eff_errup


def linregress(x, y):
    """
    Implementation of a function for linear regression
    Parameters: 2 Arrays (x- and y-values to fit)
    x and y must be of the same length
    """
    assert(len(x) == len(y))

    # Formulas from "An Introduction to Error Analysis" (Taylor)
    N = len(y)
    Delta = N * np.sum(x**2) - (np.sum(x))**2

    A = (N * np.sum(x * y) - np.sum(x) * np.sum(y)) / Delta
    B = (np.sum(x**2) * np.sum(y) - np.sum(x) * np.sum(x * y)) / Delta

    sigma_y = np.sqrt(np.sum((y - A * x - B)**2) / (N - 2)) if N > 2 else 0

    A_error = sigma_y * np.sqrt(N / Delta)
    B_error = sigma_y * np.sqrt(np.sum(x**2) / Delta)

    # f(x) = A * x + B
    return (A, A_error, B, B_error)


def linregress_weights(x,y,w):
    """
    Fit (x,y,w) to a linear function, using exact formulae for weighted linear
    regression. This code was translated from the GNU Scientific Library (GSL),
    it is an exact copy of the function gsl_fit_wlinear.
    Data points are given as vectors x and y, and weights as w. I you have std deviations for
    the data points, provide them as w = 1 / stddev**2 (i.e. weights should be inverse variances).
    """
    # compute the weighted means and weighted deviations from the means
    # wm denotes a "weighted mean", wm(f) = (sum_i w_i f_i) / (sum_i w_i)
    assert(len(x) == len(y) and len(x) == len(w))

    W = np.sum(w)
    wm_x = np.average(x,weights=w)
    wm_y = np.average(y,weights=w)
    dx = x-wm_x
    dy = y-wm_y
    wm_dx2 = np.average(dx**2,weights=w)
    wm_dxdy = np.average(dx*dy,weights=w)
    # In terms of y = A * x + B
    A = wm_dxdy / wm_dx2
    B = wm_y - wm_x*A
    A_err = np.sqrt( 1.0 / (W*wm_dx2) )
    B_err = np.sqrt( (1.0/W) * (1.0 + wm_x**2/wm_dx2) )
    cov_AB = -wm_x / (W*wm_dx2)
    # Compute chi^2 = \sum w_i (y_i - (a + b * x_i))^2
    chi2 = np.sum (w * (y-(A*x+B))**2)
    return A, A_err, B, B_err, cov_AB, chi2
