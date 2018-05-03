import numpy as np
import ROOT

def calc_TEfficiency(nPassed, nTotal):
    """Calculate efficiency using TEfficiency class from ROOT. nPassed and nTotal can be either a scalar each,
    or lists/arrays/... with the same length, where each entry correspond to a nPassed or nTotal respectively.
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

    #Create TEfficiency object and fill it
    HistPassed = ROOT.TH1D("Passed", "Passed", length, 0, length)
    HistTotal = ROOT.TH1D("Total", "Total", length, 0, length)

    for i in range(length):
        HistTotal.SetBinContent(i+1, nTotal[i])
        HistPassed.SetBinContent(i+1, nPassed[i])

    TEff = ROOT.TEfficiency(HistPassed, HistTotal)

    eff = np.asarray([ TEff.GetEfficiency(i+1) for i in range(length) ])
    eff_errlow = np.asarray([ TEff.GetEfficiencyErrorLow(i+1) for i in range(length) ])
    eff_errup = np.asarray([ TEff.GetEfficiencyErrorUp(i+1) for i in range(length) ])

    if bScalar:
        #Restore scalar behaviour
        eff = eff[0]
        eff_errlow = eff_errlow[0]
        eff_errup = eff_errup[0]

    return eff, eff_errlow, eff_errup
