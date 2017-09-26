from ValAndErr import ValAndErr as uf
from math import sqrt
"""
A class that handles efficiency calculations.
"""

def calc_efficiency(nEvents, nAll):
    eff = float(nEvents)/float(nAll)
    return uf(eff, sqrt(eff*(1.-eff)/nAll))

def eff(df, cut, prev_cut="", weight=""):
    from math import sqrt
    
    w_after = 0.
    w_before = len(df)
    
    if cut == "":
        return uf(1,0)

    if weight == "":
        if not isinstance(prev_cut, float) and not prev_cut is "":
            w_before = float(len(df.query(prev_cut, engine='python')))
        w_after = float(len(df.query(cut, engine='python')))
    else:
        print("using weight for calculation")
        if not isinstance(prev_cut, float):
            w_before = sum(df.query(prev_cut, engine='python')[weight])
        w_after = sum(df.query(cut, engine='python')[weight])
    if isinstance(prev_cut, float):
        w_before = prev_cut
    

    eff = w_after / w_before
    err = sqrt(eff*(1.-eff)/w_before)
    return uf(eff, err)
